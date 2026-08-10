#pragma once
#include <vector>
#include <cstddef>

class Tensor{
    public:
        explicit Tenors(std::vector<int> shape);
        
        float& at(std::initialier_list<int> idx);
        const float& at(std::initializer_list<int> idx) const;

        float* data();
        const float* data() const;

        const std::vector<int>& shape() const;
        size_t numel() const;
    
    private:
        std::vector<int> shape_;
        std::vector<float> data_;
};