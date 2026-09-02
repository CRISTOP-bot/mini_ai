#pragma once
#include <string>
#include <vector>
#include <cstddef>
namespace mini_ai { class ByteTokenizer { public: static constexpr size_t vocab_size=256; std::vector<int> encode(const std::string& s)const{std::vector<int> r; for(unsigned char c:s)r.push_back(c); return r;} std::string decode(const std::vector<int>& x)const{std::string s; for(int c:x)s.push_back(char(c&255)); return s;} }; }
