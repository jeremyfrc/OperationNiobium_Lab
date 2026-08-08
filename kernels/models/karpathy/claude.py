import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


def quick_print(flag=True, *args, **kwargs):
    if flag:
        print(*args, **kwargs)

with open('/home/frc28/OperationNiobium_Lab/kernels/models/karpathy/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

B, T, C = 4, 8, 32 # batch size, time steps, channels
chars = sorted(list(set(text)))
vocab_size = len(chars)
block_size = 256
n_layer = 6
n_head = 6
dropout = 0.2
n_embd = 384
max_iters = 5000


"""D1
QKV = 每个token由输入 x 经过三个线性层投影出三个向量：query， key，value。
scores = Q @ K^T， 形状(T,T):scores[i][j] = token i的query和token j 的key的相似度(点积)
除以sqrt(hs)点积随维度变大而方差爆炸 --> softmax会变得极端(几乎one-hot)，缩放以后，可以稳住方差，让权重平滑可学
GPT只看左边，所以需要一个mask，把右边j > i的scores设成-inf， softmax之后变成0。
softmax每行和为1的注意力权重。 out = weisht@V (T, hs) 每个token=它能看到的哪些token的value的加权和。
"""

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:x.shape[1], :x.shape[1]] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

"""D2
Multi-head:并行跑h个head(每个hs = C/h), 把他们输出拼接回(B,T,C)再过一个输出投影Linear。
为什么多头：每个头能学不同关系，比单头表达力强
FeedForward:逐token的两层MLP(C->4C->C + ReLU/GELU)。attention负责跨token收集信息，FFN负责收集完各自再想一想。
Residua: x + sublayer(x)。为什么加x: 给梯度一条高速路，sublayer只学增量
LayerNorm(pre-norm): 对每个token的特征做归一化，放在sublayer之前(sublayer(ln(x)))，稳定训练。
"""

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, C)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # 通信attention + 残差
        x = x + self.ffwd(self.ln2(x))  # 计算(FFN) + 残差
        return x


"""D3
token embedding + position embedding : x = tok_emb(idx) + pos_emb(arange(T)) 
为什么要position:  attention本身对顺序无感(交换token位置结果一样) --> 必须显示注入位置
堆N个Block --> final LayerNorm -> lm_head(Linear C->vocab) -> logits(B,T,vocab)。loss逻辑和bigram一样(reshape + cross_entropy)
generate: 沿用bigram的循环,但每步前要把idx裁剪到最后block_size个token, 因为pos_emb只到block_size。
self vs cross self-attention = Q/K/V全来自同一序列(GPT就是纯 self + casual mask); cross-attention = Q来自于一个序列(decoder正在生成的)、K/V来自另一个序列(encoder编码的输入) --> 我生成时该看输入的哪些词。翻译多模态用cross。
"""

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, C)
        self.position_embedding_table = nn.Embedding(block_size, C)
        self.blocks = nn.Sequential(*[Block(C, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok = self.token_embedding_table(idx) # (B, T, C)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok + pos   # 广播 -> (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        Bv, Tv, Cv = logits.shape
        loss = F.cross_entropy(logits.view(Bv*Tv, Cv), targets.view(Bv*Tv))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size] # <- 裁剪到context 窗口
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
