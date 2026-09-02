#include "mini_ai/mini_ai.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
using namespace mini_ai;
int main(int argc, char **argv) {
    std::string file = argc > 1 ? argv[1] : "../data/train.txt",
                ck = argc > 2 ? argv[2] : "mini_ai.ckpt";
    std::ifstream in(file);
    if (!in) {
        std::cerr << "cannot open " << file << "\n";
        return 1;
    }
    std::string text((std::istreambuf_iterator<char>(in)), {});
    ByteTokenizer tok;
    auto ids = tok.encode(text);
    Config c;
    Dataset data(ids, c.seq, 42);
    Model model(c);
    try {
        model.load(ck);
        std::cout << "resumed " << ck << "\n";
    } catch (...) {
        std::cout << "starting new model\n";
    }
    for (int step = 0; step < 100; step++) {
        float l = model.train_batch(data.sample(2));
        if (step % 10 == 0)
            std::cout << "step " << step << " loss " << l << "\n";
    }
    model.save(ck);
    std::cout << "saved " << ck << " (" << model.parameters() << " parameters)\n";
}
