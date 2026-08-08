import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


def quick_print(flag=True, *args, **kwargs):
    if flag:
        print(*args, **kwargs)

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

string = "length of dataset in characters:" + str(len(text))
quick_print(False, string)
quick_print(False, text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

quick_print(False, "".join(chars))
quick_print(False, vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

quick_print(False, encode("hii there"))
quick_print(False, decode(encode("hii there")))


data = torch.tensor(encode(text), dtype=torch.long)
quick_print(False, data.shape, data.dtype)
quick_print(False, data[:1000])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
quick_print(False, train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    quick_print(False, f"when input is {context} the target: {target}")

batch_size = 4

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch("train")
quick_print(False, "inputs:")
quick_print(False, xb.shape)
quick_print(False, xb)
quick_print(False, "targets:")
quick_print(False, yb.shape)
quick_print(False, yb)

print("--------------------------")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        quick_print(False, f"when input is {context.tolist()} the target: {target}")


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # B, T, C=65

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
quick_print(False, logits.shape)
quick_print(False, loss)
quick_print(False, decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(50000):
    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

quick_print(True, decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

