#include "mini_ai/mini_ai.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
using namespace mini_ai;
int main() {
    ByteTokenizer t;
    auto x = t.encode(std::string("abc\0", 4));
    assert(x.size() == 4 && t.decode(x) == "abc\0");
    Tensor a({2, 3});
    assert(a.size() == 6);
    Config c{256, 4, 8, 16};
    Model m(c);
    std::vector<int> ids = {1, 2, 3, 4, 5, 6, 7};
    Dataset d(ids, 4);
    auto b = d.sample(1);
    float l = m.train_batch(b);
    assert(std::isfinite(l));
    m.save("unit.ckpt");
    Model n(c);
    n.load("unit.ckpt");
    assert(n.parameters() == m.parameters());
    std::cout << "ok\n";
}
