import torch
import torch.nn as nn
from torch.nn import functional as F
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
sequence_len = 128  # what is the maximum context length for predictions
n_embd = 32  # embedding demension.
eval_interval = 100

# setp 1. read the text input
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"length of dataset in characters: {len(text):,d}")

# Look at the first 1000 characters
print(text[:1000])

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
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

print(ecode("hello world"))
print(decode(ecode("hello world")))

# setp 3. encode the entire text dataset and store it into a torch.Tensor

data = torch.tensor(ecode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# setp 4. split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest for evaluation
data_train, data_valid = data[:n], data[n:]

# We train the model chunk by chunk [a sequence]

print(data_train[: sequence_len + 1])

# Demo the example of the training
x = data_train[:sequence_len]
y = data_train[1 : sequence_len + 1]
for t in range(sequence_len):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = data_train if split == "train" else data_valid
    # Randomly pickup the sequence
    ix = torch.randint(len(data) - sequence_len, (batch_size,))
    x = torch.stack([data[i : i + sequence_len] for i in ix])
    y = torch.stack([data[i + 1 : i + sequence_len + 1] for i in ix])
    return x.to(device), y.to(device)


xb, yb = get_batch("train")
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

print("----")

for b in range(batch_size):  # batch dimension
    for t in range(sequence_len):  # time dimension
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(
            f"batch:[{b}], sequence_idx:[{t}], when input is {context.tolist()} the target: {target}"
        )


# Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(sequence_len, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        # (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        tok_emb += pos_emb
        logits = self.lm_head(tok_emb)  # (B, T, vocab_size)

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
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size, n_embd).to(device)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))

# Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


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

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
