import numpy as np
from pathlib import Path
D = Path("tests/ref/data")

def ld(n, s): return np.fromfile(D/n, np.float32).reshape(s)

x   = ld("attn_input.bin", (8,64))
wQ  = ld("attn_wQ.bin",    (64,64))
wK  = ld("attn_wK.bin",    (64,32))

SL, DM = 8, 64
ha, dha   = 4, 16      # n_heads, d_head
kg, dhkv  = 2, 16      # n_kv, d_head (kv head_dim same = 16)
theta = 10000.0

def cpp_rope(flat, numHeads, H, posOffset=0):
    # flat: numpy (SL, H*d_head) head-major, 精确复刻 rope_inplace
    D = dha
    out = flat.copy().astype(np.float64)
    total = flat.size
    rows = total // D          # numRows = seq_len * numHeads
    for r in range(rows):
        seqIdx = (r // numHeads) + posOffset
        row = out.reshape(-1, D)[r]
        for i in range(D // 2):
            freq = 1.0 / (theta ** (2.0*i / D))
            val  = float(seqIdx) * freq
            c = np.cos(val); s = np.sin(val)
            x0 = row[2*i]; x1 = row[2*i+1]
            row[2*i]   = x0*c - x1*s
            row[2*i+1] = x0*s + x1*c
    return out

qproj = (x @ wQ)           # (8,64) head-major
q_cpp = cpp_rope(qproj, ha, 4)

# 你的 rope_interleaved 方式
def rope_interleaved(x3d, theta):
    S,H,D = x3d.shape; DH=D//2
    i=np.arange(DH); freqs=1.0/(theta**(2.0*i/D))
    ang=np.arange(S)[:,None]*freqs[None,:]; cos=np.cos(ang); sin=np.sin(ang)
    xr=x3d.reshape(S,H,DH,2); x0=xr[...,0]; x1=xr[...,1]
    cos=np.broadcast_to(cos[:,None,:],(S,H,DH)); sin=np.broadcast_to(sin[:,None,:],(S,H,DH))
    out0=x0*cos-x1*sin; out1=x0*sin+x1*cos
    return np.stack([out0,out1],-1).reshape(S,H,D)

q_3d = qproj.reshape(SL, ha, dha)
q_my = rope_interleaved(q_3d, theta).reshape(SL, ha*dha)

print("diff q_my vs q_cpp max:", np.abs(q_my - q_cpp.reshape(SL, ha*dha)).max())



import numpy as np
from pathlib import Path
D=Path("tests/ref/data")
def ld(n,s): return np.fromfile(D/n,np.float32).reshape(s)
x=ld("attn_input.bin",(8,64)); wK=ld("attn_wK.bin",(64,32))
SL,DM=8,64; nkv,dh=2,16; theta=10000.0

def cpp_rope(flat,numHeads):  # head-major 平铺, 复刻 C++ (与之前 Q 验证用同款)
    D=dh; out=flat.copy().astype(np.float64); rows=flat.size//D
    for r in range(rows):
        seqIdx=r//numHeads; row=out.reshape(-1,D)[r]
        for i in range(D//2):
            freq=1.0/(theta**(2.0*i/D)); val=float(seqIdx)*freq
            c=np.cos(val); s=np.sin(val); x0=row[2*i]; x1=row[2*i+1]
            row[2*i]=x0*c-x1*s; row[2*i+1]=x0*s+x1*c
    return out

kproj=(x@wK)  # (8,32) head-major 2头
k_cpp=cpp_rope(kproj, nkv)

def rope_interleaved(x3d,theta):
    S,H,D=x3d.shape; DH=D//2; i=np.arange(DH)
    freqs=1.0/(theta**(2.0*i/D)); ang=np.arange(S)[:,None]*freqs[None,:]
    cos=np.cos(ang); sin=np.sin(ang)
    xr=x3d.reshape(S,H,DH,2); x0=xr[...,0]; x1=xr[...,1]
    cos=np.broadcast_to(cos[:,None,:],(S,H,DH)); sin=np.broadcast_to(sin[:,None,:],(S,H,DH))
    return np.stack([x0*cos-x1*sin,x0*sin+x1*cos],-1).reshape(S,H,D)

k_my=rope_interleaved(kproj.reshape(SL,nkv,dh),theta).reshape(SL,nkv*dh)

print("diff k_my vs k_cpp max:", np.abs(k_my - k_cpp.reshape(SL,nkv*dh)).max())
print("k_my[0,:8] =", k_my[0,:8])
print("k_cpp[0,:8]=", k_cpp[0,:8])