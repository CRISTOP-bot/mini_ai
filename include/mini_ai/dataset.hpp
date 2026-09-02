#pragma once
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
namespace mini_ai { struct Batch { std::vector<int> x,y; std::size_t batch=0,seq=0; };
class Dataset { std::vector<int> ids_; std::size_t seq_; std::mt19937 rng_; public:
 Dataset(std::vector<int> ids,std::size_t seq,unsigned seed=1):ids_(std::move(ids)),seq_(seq),rng_(seed){if(seq_==0||ids_.size()<seq_+1)throw std::invalid_argument("dataset too short");}
 std::size_t size()const{return ids_.size();}
 Batch sample(std::size_t batch){Batch b; b.batch=batch;b.seq=seq_; std::uniform_int_distribution<std::size_t>d(0,ids_.size()-seq_-1); for(std::size_t n=0;n<batch;n++){auto p=d(rng_);for(std::size_t t=0;t<seq_;t++){b.x.push_back(ids_[p+t]);b.y.push_back(ids_[p+t+1]);}}return b;}
}; }