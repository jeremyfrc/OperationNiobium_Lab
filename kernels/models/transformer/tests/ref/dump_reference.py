import os
import torch
import torch.nn as nn
import numpy as np


torch.manual_seed(42)
np.random.seed(42)

tiny_cfg = {
    'hidden_dim': 64,
    'num_heads': 4,
    'num_kv_heads': 2,
    'head_dim': 16,
    'ffn_dim': 128,
    'vocab_size': 256,
    'eps': 1e-5
}

REF_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(REF_DIR, exist_ok=True)

def rand(*s):
    return np.random.randn(*s).astype(np.float32)

def save_bin(name, tenosr_or_array):
    """将 pytorch tensor保存为连续的float32 二进制文件"""
    if isinstance(tenosr_or_array, torch.Tensor):
        arr = tenosr_or_array.detach().cpu().numpy().astype(np.float32)
    else:
        arr = tenosr_or_array.astype(np.float32)
    path = os.path.join(REF_DIR, name)
    arr.tofile(path)
    print(f"Saved {name}: shape={arr.shape}")

def rmsnorm_ref(x: np.ndarray, w: np.ndarray, eps : float=1e-5) -> np.ndarray:
    """
    x: shape(...,dim), float 32
    w: shape(dim,), float 32
    """
    ms = np.mean(x**2, axis=-1, keepdims=True)
    
    return (x / np.sqrt(ms + eps) * w).astype(np.float32)

def softmax_ref(x: np.ndarray):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(np.float32)


def rope_ref(x: np.ndarray, theta: float=10000.0) -> np.ndarray:
    """
    x: shape (batch_size, seq_len, num_heads, head_dim)
    head_dim must be even
    """
    x = np.asanyarray(x, dtype=np.float32)
    B, S, H, D = x.shape
    assert D % 2 == 0, "head_dim must be even!"

    half_D = D // 2
    # 1. 计算positions: [0, 1, ..., S-1] -> shape(S, 1)
    pos = np.arange(S, dtype=np.float32)[:, np.newaxis]

    # 2. 计算frequencies: 1.0 / (theta ** (2i/D)) -> shape(D/2,)
    i = np.arange(half_D, dtype=np.float32)
    freqs = 1.0 / (theta ** (2.0 * i / D))

    # 3. 外积计算角度angles: shape(S, D/2)
    angles = pos * freqs

    # 4. 广播cos/sin 形状至(1, S, 1, D/2)
    cos = np.cos(angles)[np.newaxis, :, np.newaxis, :]
    sin = np.sin(angles)[np.newaxis, :, np.newaxis, :]

    # 5. 拆分x的前半部分和后半部分
    x1 = x[..., :half_D]
    x2 = x[..., half_D:]

    # 6. 旋转公式： [x1, x2] * cos + [-x2, x1] * sin
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return np.concatenate([out1, out2], axis=-1).astype(np.float32)


def swiglu_ref(gate: np.ndarray, up: np.ndarray):
    silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
    return (silu_gate * up).astype(np.float32)

# ==========================================
# 1. Softmax Reference Data
# ==========================================
x_sm = rand(4, 16) * 10.0
out_sm = softmax_ref(x_sm)

save_bin("softmax_input.bin", x_sm)
save_bin("softmax_out.bin", out_sm)

# ==========================================
# 2. RMSNorm Reference Data
# ==========================================
dim, eps = 64, 1e-5
x_rms = rand(2, 16, 64)
w_rms = rand(64) + 1.0
out_rms = rmsnorm_ref(x_rms, w_rms)

save_bin("rmsnorm_input.bin", x_rms)
save_bin("rmsnorm_weight.bin", w_rms)
save_bin("rmsnorm_out.bin", out_rms)

# ==========================================
# 3. RoPE Reference Data
# ==========================================
x_rope = rand(1, 8, 2, 16) # [batch=1, seq_len=8, num_heads=2, head_dim=16]
out_rope = rope_ref(x_rope)

save_bin("rope_input.bin", x_rope)
save_bin("rope_out.bin", out_rope)

# ==========================================
# 4. SwiGLU FFN Reference Data
# ==========================================
gate_in = rand(2, 8, 32)
up_in = rand(2, 8, 32)
out_swiglu = swiglu_ref(gate_in, up_in)

save_bin("swiglu_gate_in.bin", gate_in)
save_bin("swiglu_up_in.bin", up_in)
save_bin("swiglu_out.bin", out_swiglu)


print("\n🎉 Stage 1 所有的 Reference 数据已经全部成功导出至 tests/ref/data/ 目录！")
