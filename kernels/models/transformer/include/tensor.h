#pragma once
#include <vector>
#include <cstddef>

class Tensor{
    public:
        explicit Tensor(std::vector<int> shape);

        Tensor(std::vector<int> shape, const std::vector<float>& data);

        float& at(std::initializer_list<int> idx);
        const float& at(std::initializer_list<int> idx) const;

        float* data();
        const float* data() const;

        const std::vector<int>& shape() const;
        size_t numel() const;

        //原地改变逻辑形状；数据连续，仅重算strides_
        void reshape(std::vector<int> new_shape);
    
    private:
        std::vector<int> shape_;
        std::vector<float> data_;
        std::vector<int> strides_;
};