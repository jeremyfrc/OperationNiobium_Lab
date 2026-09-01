#pragma once
#include <vector>
#include <cstddef>

class Tensor{
    public:
        explicit Tensor(std::vector<int> shape);
        Tensor(std::vector<int> shape, std::vector<float> data): shape_(std::move(shape)), data_(std::move(data)) {}
        
        float& at(std::initializer_list<int> idx);
        const float& at(std::initializer_list<int> idx) const;

        float* data();
        const float* data() const;

        const std::vector<int>& shape() const;
        size_t numel() const;
    
    private:
        std::vector<int> shape_;
        std::vector<float> data_;
        std::vector<int> strides_;
};