#pragma once
#include "dataset.hpp"
#include "tensor.hpp"

#include <cstddef>
#include <string>
#include <vector>
namespace mini_ai {
struct Config {
    std::size_t vocab = 256, seq = 32, d_model = 32, d_ff = 64;
};
class Adam;
class Model {
    Config c_;
    std::vector<Tensor> p_, g_;
    Adam *adam_;
    std::size_t steps_ = 0;

  public:
    explicit Model(Config c = Config{});
    ~Model();
    float train_batch(const Batch &);
    std::vector<float> logits(const std::vector<int> &);
    std::vector<int> generate(std::vector<int> ids, std::size_t n, float temperature = 1);
    void save(const std::string &) const;
    void load(const std::string &);
    std::size_t parameters() const;
    const Config &config() const {
        return c_;
    }
    std::size_t steps() const {
        return steps_;
    }

  private:
    void init();
    void update();
};
} // namespace mini_ai