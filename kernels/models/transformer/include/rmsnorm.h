#pragma once
#include "tensor.h"

// out = x / rms(x, 沿最后一维) * weight
// rms(x) = sqrt(mean(x^2) + eps)
void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out, float eps = 1e-5f);