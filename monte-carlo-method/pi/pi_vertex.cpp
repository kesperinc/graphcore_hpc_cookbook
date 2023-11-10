// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <ipudef.h>
#include <climits>
#include <print.h>
#include <math.h>

using namespace poplar;

class PiVertex : public MultiVertex {

public:
    Output<Vector<unsigned int>> hits;
    int iterations, NumElemsPerTile;

    auto compute(unsigned i) -> bool {
        int count = 0;
        for (auto i = 0; i < 50; i++) {
            // auto x = (float)__builtin_ipu_urand32() / (float)UINT_MAX;
            // auto y = (float)__builtin_ipu_urand32() / (float)UINT_MAX;
            // auto val = x * x + y * y;
            // count +=  val < 1.f; 
            hits[i] = __builtin_ipu_urand32() ;
        }
        hits[i] = __builtin_ipu_urand32() ;

        return true;
    }
    auto ipu_genrand_uint32(){
        auto x = (float)__builtin_ipu_urand32() / (float)UINT32_MAX ; //[0,2^32-1]
        return x; 
    }
    auto ipu_genrand_double(){
        auto x = (double)__builtin_ipu_urand64() / (double)ULLONG_MAX ;  //[0,2^64-1]
        return x; 
    }
    auto ipu_init_genrand(uint32_t seed){
        

    }
};

