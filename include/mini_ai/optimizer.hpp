#pragma once
#include "tensor.hpp"
#include <cstddef>
#include <iosfwd>
#include <vector>
namespace mini_ai {
class Adam {
 std::vector<Tensor> m_,v_; std::size_t t_=0;
public:
 void step(std::vector<Tensor>& p,const std::vector<Tensor>& g,float lr=1e-3f,float b1=.9f,float b2=.999f,float eps=1e-8f);
 void clear(){m_.clear();v_.clear();t_=0;}
 std::size_t step_count() const { return t_; }
 void save(std::ostream&) const; void load(std::istream&,const std::vector<Tensor>&);
};
}