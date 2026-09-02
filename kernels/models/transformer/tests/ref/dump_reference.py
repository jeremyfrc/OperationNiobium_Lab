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
    pos = np.arange(S, dtype=np.float32)[:, np.newaxis] # (S, 1)
    
    # 频率: shape (D/2,)
    i = np.arange(half_D, dtype=np.float32)
    freqs = 1.0 / (theta ** (2.0 * i / D))

    # angles: shape (S, D/2)
    angles = pos * freqs 

    # cos/sin: shape (1, S, 1, D/2)
    cos = np.cos(angles)[np.newaxis, :, np.newaxis, :]
    sin = np.sin(angles)[np.newaxis, :, np.newaxis, :]

    # 适配 C++ 的交错式 (interleaved) 旋转:
    # x 对应形状 [B, S, H, D]，我们将最后一维拆成 [..., half_D, 2]
    x_reshaped = x.reshape(B, S, H, half_D, 2)
    x0 = x_reshaped[..., 0] # 前半交错项
    x1 = x_reshaped[..., 1] # 后半交错项

    # 扩展 cos/sin 形状以匹配广播 [B, S, H, half_D]
    cos = np.broadcast_to(cos, (B, S, H, half_D))
    sin = np.broadcast_to(sin, (B, S, H, half_D))

    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    # 重新拼回 [B, S, H, D]
    out = np.stack([out0, out1], axis=-1)
    return out.reshape(B, S, H, D).astype(np.float32)


def ffn_swiglu_ref(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    """
    x:       shape (batch, seq_len, in_dim)
    w_gate:  shape (in_dim, hidden_dim)       # 直接存 (in_dim, intermediate_dim)
    w_up:    shape (in_dim, hidden_dim)       # 直接存 (in_dim, intermediate_dim)
    w_down:  shape (hidden_dim, in_dim)       # 直接存 (intermediate_dim, in_dim)
    """
    gate = np.matmul(x, w_gate)  # 不再转置
    up   = np.matmul(x, w_up)    # 不再转置
    
    silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
    hidden = silu_gate * up
    
    out = np.matmul(hidden, w_down) # 不再转置
    return out.astype(np.float32)

# ==========================================
# 1. Softmax Reference Data
# ==========================================
x_sm = rand(2, 4, 8, 8)
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
batch, seq_len = 2, 4
in_dim = 64        # hidden_dim
hidden_dim = 128   # intermediate_dim / ffn_dim
num_tokens = batch * seq_len

# 2. 随机生成符合 B 方案 Shapes 的输入与权重
# 输入 x: (batch, seq_len, in_dim)
x = np.random.randn(batch, seq_len, in_dim).astype(np.float32)
    
# B 方案：权重形状直接存为 C++ matmul 期望的 [in, out]
w_gate = np.random.randn(in_dim, hidden_dim).astype(np.float32)
w_up   = np.random.randn(in_dim, hidden_dim).astype(np.float32)
w_down = np.random.randn(hidden_dim, in_dim).astype(np.float32)

# 3. 使用前向计算得到 reference output
out = ffn_swiglu_ref(x, w_gate, w_up, w_down)

# 4. 直接保存二进制文件 (无需任何 .T 转置)
save_bin("swiglu_x.bin", x)
save_bin("swiglu_w_gate.bin", w_gate)
save_bin("swiglu_w_up.bin", w_up)
save_bin("swiglu_w_down.bin", w_down)
save_bin("swiglu_out.bin", out)

print("\n🎉 Stage 1 所有的 Reference 数据已经全部成功导出至 tests/ref/data/ 目录！")
