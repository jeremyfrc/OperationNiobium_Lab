#pragma once
#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>

// 1. loader: load .bin file to std::vector<float>
inline std::vector<float> loadBinFile(const std::string& filepath){
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()){
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read binary data from: " + filepath);
    }

    return buffer;
}


// 2. Checker: compare C++ and PYthon ref results
inline bool check_close(const float* actual, const float* expected, size_t size, float rtol = 1e-4f, float atol = 1e-5f) {
    size_t errors = 0;
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(actual[i] - expected[i]);
        float max_error = atol + rtol * std::abs(expected[i]);

        if (diff > max_error) {
            if (errors < 5) {
                std::cerr << "Mismatch at index " << i 
                << "; actual= " << actual[i] 
                << ", expected=" << expected[i] 
                << ", diff=" << diff << std::endl;
            }
            errors++;
        }
    }

    if (errors > 0){
        std::cerr << "Total mismatches: " << errors << " /  " << size << std::endl;
        return false;
    }
    return true;
}