import numpy as np
from pathlib import Path

D = Path("tests/ref/data")

def ld(n, s):
    return np.fromfile(D / n, dtype=np.float32).reshape(s).astype(np.float64)

x   = ld("attn_input.bin", (8, 64))
wQ  = ld("attn_wQ.bin",    (64, 64))
wK  = ld("attn_wK.bin",    (64, 32))
wV  = ld("attn_wV.bin",    (64, 32))
wO  = ld("attn_wO.bin",    (64, 64))
ref = ld("attn_out.bin",   (8, 64))

h, kvg, dh, sl = 4, 2, 16, 8
group = h // kvg

q = x @ wQ
k = x @ wK
v = x @ wV

scores = np.full((h, sl, sl), -1e9)
for hh in range(h):
    kv = hh // group
    for i in range(sl):
        for j in range(i + 1):
            scores[hh, i, j] = (q[i, hh*dh:(hh+1)*dh] @ k[j, kv*dh:(kv+1)*dh]) / np.sqrt(dh)

m = scores.max(-1, keepdims=True)
e = np.exp(scores - m)
probs = e / e.sum(-1, keepdims=True)

ctx = np.zeros((h, sl, dh))
for hh in range(h):
    kv = hh // group
    for i in range(sl):
        for d in range(dh):
            ctx[hh, i, d] = sum(probs[hh, i, j] * v[j, kv*dh + d] for j in range(i + 1))

ctx = ctx.transpose(1, 0, 2).reshape(sl, h * dh)
out = ctx @ wO

print("my out[0] =", out[0])
print()
print("ref[0]    =", ref[0])
print()
print("max abs diff:", np.abs(out - ref).max())