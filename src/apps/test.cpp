#include "mini_ai/mini_ai.hpp"
#include <iostream>
using namespace mini_ai;
int main(int argc,char**argv){Config c;Model m(c);if(argc>1)try{m.load(argv[1]);}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}ByteTokenizer t;std::string prompt=argc>2?argv[2]:"Hello";auto ids=t.encode(prompt);auto out=m.generate(ids,80,.8f);std::cout<<t.decode(out)<<"\n";}
