import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from tqdm import tqdm
from datetime import datetime
import math
from tensorboardX import SummaryWriter

# set random seed
torch.manual_seed(1337)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Config
epochs = 1_0000
batch_size = 64  # how many independent sequences will we process in parallel
sequence_len = 256  # what is the maximum context length for predictions
n_embd = 384  # embedding demension.
n_blocks = 6
n_head = 6
dropout = 0.2
eval_interval = 100
log_interval = 10
learning_rate = 3e-4

# Initialize TensorBoardX writer
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(logdir=f"runs/experiment_{run_id}")

# setp 1. read the text input
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"length of dataset in characters: {len(text):,d}")

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"length of vocab: {vocab_size:,d}")

# setp 2. tokenizing, we use the most simplest approach: char index
# others like
# - https://github.com/google/sentencepiece
# - https://github.com/openai/tiktoken
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# endoder: take a string, output a list of intergers
ecode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers, output a string
decode = lambda l: "".join([itos[i] for i in l])

# setp 3. encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(ecode(text), dtype=torch.long)

# setp 4. split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest for evaluation
data_train, data_valid = data[:n], data[n:]


# We train the model chunk by chunk [a sequence]
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = data_train if split == "train" else data_valid
    # Randomly pickup the sequence
    ix = torch.randint(len(data) - sequence_len, (batch_size,))
    x = torch.stack([data[i : i + sequence_len] for i in ix])
    y = torch.stack([data[i + 1 : i + sequence_len + 1] for i in ix])
    return x.to(device), y.to(device)


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        # What information should this token focus on?
        self.key = nn.Linear(n_embd, head_size, bias=False)
        # What does this token represent?
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # What information does this token contain?
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(sequence_len, sequence_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        # Attention scores represents the similarity between Q and K (how relevant is a token?)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel
    It allow the token able to
    1. talk to other tokens in diferrent ways
    2. learn different types of relationships between tokens
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity
    Intuition
    1. Think of self-attention as handling global interactions between tokens
    2. Transforms each token’s representation in a non-linear way, enriching its features.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        """
        Adding layer normal to avoid over fitting
        It ensures that the activations have a mean of 0 and a variance of 1 across the feature dimension for each sample,
        making the optimization process smoother.
        1. Works Well for Small Batch Sizes
        LayerNorm works independently for each input, making it ideal for NLP, Transformers, and RL
        2.Stable Training
        Normalizing across features prevents large activation shifts, improving convergence.
        3. LayerNorm before/after attention layers to stabilize gradients.
        Cons, Slightly higher
        layer normal https://arxiv.org/abs/1607.06450
        """
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        # residual connect
        # https://medium.com/towards-data-science/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal encodings allow Transformers to understand order without recurrence.
    --------------------------------------------------------------------------------------------------------------------------
    There are two main approaches of position embedding,
    1. Learned Positional Embeddings
       - Each position gets an embedding vector (like token embeddings).
       - Pros: Simple, effective.
       - Cons: Limited to training sequence lengths.
               If you train a model with sequence_len = 256,
               the position embeddings are trained only for indices 0 to 255.
               If you try to infer with a longer sequence (e.g., 512 tokens),
               there's no learned embedding for those new positions → the model fails to generalize.
    2. Sinusoidal Positional Encoding
       - Uses sine and cosine functions to generate a continuous representation for each position.
       - Pros: Generalizes well, supports arbitrary sequence lengths.
       - Cons: More complex.

    📌 If your model always works with fixed-length sequences (e.g., BERT trained on 512 tokens), learned embeddings work fine.
    However, for arbitrary-length sequences (e.g., GPT-style autoregressive generation), sinusoidal encodings are preferable.
    --------------------------------------------------------------------------------------------------------------------------
    🚀 Why Sinusoidal Works
    1. Handles arbitrary sequence lengths:
       Unlike learned embeddings, this method generalizes beyond training data.
       Generalizes to any sequence length:
         Because it's a continuous function, it can be evaluated at any position,
         even those not seen during training.
    2. Encodes absolute & relative positions:
        Different positions have different encodings, helping the model understand order.
    3. Smooth transitions:
       The sine/cosine functions ensure gradual changes between positions.
    --------------------------------------------------------------------------------------------------------------------------
    🚀 Why Use the Division:
    1. 10000/2^(i/d) controls frequency scaling
       - Each dimension of the positional encoding represents a different wavelength of a sine/cosine wave.
       - The denominator scales down the position index.
       - As i (the embedding dimension index) increases, the frequency of the sine wave decreases.
       - Higher dimensions capture broader positional differences (long-range dependencies).
       - Lower dimensions capture finer-grained differences (short-range dependencies).
    2. Why use an exponential scale?
       If we just used sin(pos), the frequency would be too high, and small position changes would result in large oscillations.
       Instead, we divide by 10000^2i/d so that
       - For small values of i, the frequency is high, capturing fine-grained token positions.
       - For large values of i, the frequency is low, capturing long-range dependencies.
    --------------------------------------------------------------------------------------------------------------------------
    🚀 Why Use Separate Sine and Cosine?
    We use sine for even indices and cosine for odd indices to
    - Ensure orthogonality between different dimensions.
    - Allow for better expressiveness in encoding positions.
    - Make the position encoding unique and invertible (you can recover the position from the encoding).
    --------------------------------------------------------------------------------------------------------------------------
    ✅ Example: How It Works
    Let’s take d = 4 (4-dimensional embeddings) and calculate the encoding for pos = 1, 2, 3
    |pos | PE(0) = sin	   | PE(1) = cos	 | PE(2) = sin	        | PE(3) = cos          |
    |1	 | sin(1 / 10000⁰) | cos(1 / 10000⁰) | sin(1 / 10000^(2/4)) | cos(1 / 10000^(2/4)) |
    |2	 | sin(2 / 10000⁰) | cos(2 / 10000⁰) | sin(2 / 10000^(2/4)) | cos(2 / 10000^(2/4)) |
    |3	 | sin(3 / 10000⁰) | cos(3 / 10000⁰) | sin(3 / 10000^(2/4)) | cos(3 / 10000^(2/4)) |
    --------------------------------------------------------------------------------------------------------------------------
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Adds positional encoding to input embeddings."""
        return x + self.pe[:, : x.size(1), :]


# Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head=4):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = SinusoidalPositionalEncoding(
            n_embd, max_len=sequence_len
        )
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        """
        🚀 Token Embeddings (tok_emb)
        - Token embeddings are the learned representations of the individual tokens (words, characters, or subwords) in your vocabulary.
        - When a token (such as a word or subword) is input into the model, its index in the vocabulary is used to look up the corresponding embedding.
        - These embeddings capture semantic meaning. For instance, the token "cat" might have an embedding that contains information about 
          the object "cat", similar to how "dog" is represented but distinct from "fish."
        In essence, token embeddings give you a representation of the meaning of the token in the context of the vocabulary.
        """
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        # (T, C)
        """
        🚀 Positional Embeddings (pos_emb)
        - Positional embeddings are designed to give the model information about the position of each token in the sequence. 
          This is crucial because, unlike RNNs or CNNs, Transformers don’t inherently process tokens sequentially or in any particular order.
        - The positional encoding (often created using a sinusoidal function) allows the model to distinguish the position of 
          a token within the sequence.
        - The positional encoding helps the Transformer model understand which token is first, second, third, etc., in the sequence.
        Without positional embeddings, the model wouldn't know the order of the tokens (whether "dog eats cat" is different from "cat eats dog").
        """
        pos_emb = self.positional_encoding(
            tok_emb
        )  # Apply sinusoidal position encoding
        """
        🚀 Summary
        - Token embeddings give the semantic meaning of tokens.
        - Positional embeddings give the relative positions of the tokens in the sequence.
        - By adding them together, we create a combined representation that captures both meaning and order.
        - This combined embedding is then fed into the model, enabling it to learn relationships between the tokens and their positions in the sequence.
        
        🚀 Why Adding the Embeddings Works
        The key point here is that adding token embeddings and positional embeddings allows the model to 
        leverage both the semantic meaning of the tokens and the position in the sequence simultaneously
        - Linear Combination of Information. 
          By adding the embeddings, you combine the content (from the token embeddings) with the position (from the positional embeddings)
          This combination of information is what enables the model to capture both meaning and context (sequence order) simultaneously.
        - No Need for Explicit Sequence Handling
          Transformers do not have any inherent understanding of sequence order, unlike RNNs or LSTMs. 
          So, by adding positional embeddings to token embeddings, we ensure that each token has information about:
          - Its identity (what it is, thanks to token embeddings), and
          - Its position (where it is in the sequence, thanks to positional embeddings).
        """
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # calculate the loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # reshape the logits
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last sequence size tokens
            idx_cond = idx[:, -sequence_len:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size, n_embd, n_head).to(device)

fake_input = torch.randint(
    0, vocab_size, (batch_size, sequence_len), dtype=torch.long
).to(device)
print(summary(model, input_data=fake_input))
print("---------------------------------")

# Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.inference_mode(True)
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "valid"]:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Early stopping parameters
patience = 5  # Number of epochs to wait for improvement before stopping
best_val_loss = float("inf")  # Initialize best validation loss to infinity
epochs_without_improvement = 0  # Counter for epochs without improvement

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=3, factor=0.5
)

status = tqdm(range(epochs), desc="Training", unit="epoch")
for epoch in status:
    current_lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("Learning Rate", current_lr, epoch)

    # sample a batch of data
    xb, yb = get_batch("train")

    if epoch % eval_interval == 0:
        losses = estimate_loss()
        status.set_postfix_str(
            f"loss:{losses['train']:.4f}, val_loss:{losses['valid']:.4f}"
        )

        # Log the losses to TensorBoard
        writer.add_scalar("Loss/train", losses["train"], epoch)
        writer.add_scalar("Loss/val", losses["valid"], epoch)

        # Log generated samples
        generated_text = decode(model.generate(xb, max_new_tokens=100)[0].tolist())
        writer.add_text("Generated Text", generated_text, epoch)

        val_loss = losses["valid"]
        # Update the learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping check
        # If the validation loss improved, reset the counter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # If no improvement for 'patience' epochs, stop training early
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if epoch % log_interval == 0:
        # Log gradients and weights
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, epoch)
            writer.add_histogram(f"grads/{name}", param.grad, epoch)

writer.add_scalar("Loss/final", loss.item(), epoch)
print(f"Final loss: {loss.item()}")

# Test generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# Close the TensorBoard writer
writer.close()
