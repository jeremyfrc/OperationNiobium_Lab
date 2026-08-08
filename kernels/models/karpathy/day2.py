import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


def quick_print(flag=True, *args, **kwargs):
    if flag:
        print(*args, **kwargs)

# with open('input.txt', 'r', encoding='utf-8') as file:
#     text = file.read()

print_flag = False

torch.manual_seed(1337)
B, T, C = 4, 8, 32 # batch size, time steps, channels
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # B, T, 16
q = query(x) # B, T, 16

wei = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros(T, T)
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)

#out = wei @ x
v = value(x)
out = wei @ v

quick_print(print_flag, "tril:", tril)
quick_print(print_flag, "wei:", wei)
quick_print(print_flag, "out:", out)
quick_print(print_flag, "v: ", v)


wei = q @ k.transpose(-2, -1) * head_size**-0.5
print(k.var())
print(q.var())
print(wei.var())

print(torch.softmax(torch.tensor([0.1, -0.2, -0.3, -0.2, 0.5]), dim=-1))

print(torch.softmax(torch.tensor([0.1, -0.2, -0.3, -0.2, 0.5]) * 8, dim=-1))






















