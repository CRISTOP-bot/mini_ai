#pragma once
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
namespace mini_ai {
class Tensor {
    std::vector<size_t> shape_;
    std::vector<float> data_;

  public:
    Tensor() = default;
    explicit Tensor(std::vector<size_t> s, float v = 0)
        : shape_(std::move(s)),
          data_(std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>()),
                v) {};
    static Tensor zeros(std::vector<size_t> s) {
        return Tensor(std::move(s));
    }
    size_t size() const {
        return data_.size();
    }
    const std::vector<size_t> &shape() const {
        return shape_;
    }
    float *data() {
        return data_.data();
    }
    const float *data() const {
        return data_.data();
    }
    float &operator[](size_t i) {
        return data_.at(i);
    }
    const float &operator[](size_t i) const {
        return data_.at(i);
    }
    size_t index(std::initializer_list<size_t> x) const {
        if (x.size() != shape_.size())
            throw std::invalid_argument("rank");
        size_t i = 0, k = 0;
        for (auto v : x)
            i = i * shape_[k++] + v;
        return i;
    }
};
} // namespace mini_ai
