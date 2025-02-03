from sympy import im
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from tqdm import tqdm

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
learning_rate = 3e-4

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
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
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
        # add layer normal to avoid over fitting
        # It ensures that the activations have a mean of 0 and a variance of 1 across the feature dimension for each sample,
        # making the optimization process smoother.
        # 1. Works Well for Small Batch Sizes
        # LayerNorm works independently for each input, making it ideal for NLP, Transformers, and RL
        # 2.Stable Training
        # Normalizing across features prevents large activation shifts, improving convergence.
        # 3. LayerNorm before/after attention layers to stabilize gradients.
        # Cons, Slightly higher
        # layer normal https://arxiv.org/abs/1607.06450

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


# Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head=4):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(sequence_len, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        # (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
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


status = tqdm(range(epochs), desc="Training", unit="epoch")
for epoch in status:
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        status.set_postfix_str(
            f"loss:{losses['train']:.4f}, val_loss:{losses['valid']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"loss:{loss.item()}")

# Test generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
