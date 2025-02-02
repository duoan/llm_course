from regex import W
import torch
from torch import nn

torch.manual_seed(1377)

B, T, C = 4, 8, 32  # batch, time, channels [vocab size]
x = torch.randn(B, T, C)
print(x.shape)

# Create relitionship between current token to previous tokens.
# the rough ideas is to take all previous tokens embedding features average values adding to current token.

# Version#1
# We want x[b, t] = mean(x[b, t], x[b, t+1])
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]
        xbow[b, t] = torch.mean(xprev, 0)

print(x[0])
print(xbow[0])


a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()

c = a @ b
print("a=")
print(a)
print("--")
print("b=")
print(b)
print("--")
print("c=")
print(c)


# Version#2
weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(1, keepdim=True)
xbow2 = weights @ x  # (T, T) @ (B, T, C) -> (B, T, C)
print(xbow2.shape)
print(torch.allclose(xbow, xbow2))

print(xbow[0])
print(xbow2[0])

# Version#3, use softmax
tri = torch.tril(torch.ones(T, T))
print(tri)

weights = torch.zeros((T, T))
weights = weights.masked_fill(tri == 0, float("-inf"))
print(weights)
weights = torch.softmax(weights, dim=-1)
print(weights)
xbow3 = weights @ x
print(torch.allclose(xbow, xbow3))

"""

Above implementation, mix the current token and previous token information together.
The problem is all previous tokens have same weights.
In reality, some tokens is more important or relivant to current one than others.
That's the self-attention come from.

For a single token, it comes out a query and a key, 
the query is "what should I focus on or what am I looking for?"
the key is "What I am or what I contain?"

If the query and the key are similar, so current token can know how many of weights from their focused tokens.
The similarity can be calculated by [query dot key],


"""

B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# single head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)  # the token self identity

k = key(x)  # (B, T, 16)
q = query(x)  # (B, T, 16)
weights = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
# normalize weights, to avoid softmax converge to a highest number
weights = weights / weights.shape[-1] ** 0.5

tril = torch.tril(torch.ones(T, T))
# use for decoder which don't allow current token talk to the forward tokens
weights = weights.masked_fill(tril == 0, float("-inf"))
weights = torch.softmax(weights, dim=-1)
print(weights)

v = value(x)
out = weights @ v

print(out.shape)
print(out)

"""
Attention is a communication mechanism. 
"""
