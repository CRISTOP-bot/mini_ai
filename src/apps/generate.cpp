#include "mini_ai/mini_ai.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
struct Options {
    std::string checkpoint = "mini_ai.ckpt";
    std::string prompt = "Hello";
    std::size_t tokens = 32;
    float temperature = 1.0f;
};

void usage(std::ostream &out) {
    out << "Usage: mini_ai_generate [options] [prompt]\n\n"
        << "Generate bytes from a MAI3 checkpoint.\n\n"
        << "Options:\n"
        << "  -c, --checkpoint PATH  checkpoint to load (default: mini_ai.ckpt)\n"
        << "  -t, --tokens N         number of new tokens (default: 32)\n"
        << "      --max-tokens N     alias for --tokens\n"
        << "      --temperature F    sampling temperature > 0 (default: 1.0)\n"
        << "  -h, --help             show this help\n\n"
        << "The optional prompt is tokenized as UTF-8 bytes (not Unicode code points).\n";
}

std::string require_value(int &i, int argc, char **argv, std::string_view option) {
    if (i + 1 >= argc)
        throw std::invalid_argument(std::string(option) + " requires a value");
    return argv[++i];
}

std::size_t parse_size(const std::string &text, std::string_view option) {
    if (text.empty() || text.front() == '-')
        throw std::invalid_argument(std::string(option) + " must be a non-negative integer");
    std::size_t used = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &used);
    } catch (...) {
        throw std::invalid_argument(std::string(option) + " must be a non-negative integer: " + text);
    }
    if (used != text.size() || value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(std::string(option) + " must be a non-negative integer: " + text);
    return static_cast<std::size_t>(value);
}

float parse_temperature(const std::string &text) {
    std::size_t used = 0;
    float value = 0;
    try {
        value = std::stof(text, &used);
    } catch (...) {
        throw std::invalid_argument("--temperature must be a finite number > 0: " + text);
    }
    if (used != text.size() || !std::isfinite(value) || value <= 0)
        throw std::invalid_argument("--temperature must be a finite number > 0: " + text);
    return value;
}

Options parse(int argc, char **argv) {
    Options options;
    bool prompt_seen = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(std::cout);
            std::exit(0);
        } else if (arg == "-c" || arg == "--checkpoint") {
            options.checkpoint = require_value(i, argc, argv, arg);
        } else if (arg == "-t" || arg == "--tokens" || arg == "--max-tokens") {
            options.tokens = parse_size(require_value(i, argc, argv, arg), arg);
        } else if (arg == "--temperature") {
            options.temperature = parse_temperature(require_value(i, argc, argv, arg));
        } else if (!arg.empty() && arg.front() == '-') {
            throw std::invalid_argument("unknown option: " + arg);
        } else if (prompt_seen) {
            throw std::invalid_argument("only one prompt argument is allowed");
        } else {
            options.prompt = arg;
            prompt_seen = true;
        }
    }
    if (options.prompt.empty())
        throw std::invalid_argument("prompt must not be empty");
    return options;
}
} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse(argc, argv);
        mini_ai::ByteTokenizer tokenizer;
        const auto prompt = tokenizer.encode(options.prompt);
        mini_ai::Model model;
        model.load(options.checkpoint);
        // Model::generate retains at most Config::seq tokens, so its result is
        // directly printable and includes the prompt's retained suffix.
        const auto output = model.generate(prompt, options.tokens, options.temperature);
        std::cout << tokenizer.decode(output);
        if (std::cout.good())
            std::cout << '\n';
        return std::cout.good() ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "mini_ai_generate: " << error.what() << '\n';
        return 2;
    }
}
