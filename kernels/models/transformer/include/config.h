#pragma once

struct TransformerConfig{
    int vocab_size;
    int d_model;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int d_head;
    int d_ff;
    int max_seq_len;
    float rope_theta;
    float rms_eps;
};