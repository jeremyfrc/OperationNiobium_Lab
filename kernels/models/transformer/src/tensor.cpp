#include "tensor.h"
#include <numeric>
#include <stdexcept>

Tensor::Tensor(std::vector<int> shape) : shape_(std::move(shape)) {
    int rank = shape_.size();
    strides_.resize(rank);

    size_t total_elements = 1;
    if (rank > 0){
        int current_stride = 1;
        for (int i = rank - 1; i >= 0; --i){
            strides_[i] = current_stride;
            current_stride *= shape_[i];
        }
        total_elements = current_stride;
    } else {
        total_elements = 0;
    }

    data_.resize(total_elements, 0.0f);
}

size_t Tensor::numel() const {
    return data_.size();
}

float* Tensor::data() {
    return data_.data();
}

const float* Tensor::data() const {
    return data_.data();
}

const std::vector<int>& Tensor::shape() const {
    return shape_;
}

float& Tensor::at(std::initializer_list<int> idx){
    size_t flat_idx = 0;
    auto it = idx.begin();
    for (size_t i = 0; i < idx.size(); ++i){
        flat_idx += (*it) * strides_[i];
        ++it;
    }
    return data_[flat_idx];
}

const float& Tensor::at(std::initializer_list<int> idx) const {
    size_t flat_idx = 0;
    auto it = idx.begin();
    for (size_t i = 0; i < idx.size(); ++i) {
        flat_idx += (*it) * strides_[i];
        ++it;
    }
    return data_[flat_idx];
}