// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <ipudef.h>
#include <climits>
#include <print.h>
#include <math.h>

#define PI 3.141592654
#define ran1(x) ( (float)__builtin_ipu_urand32() / (float)UINT_MAX )  

using namespace poplar;

class PiVertex : public MultiVertex {

public:
    Output<Vector<float>> prng;
    Input<Vector<float>> flag;

    float gammln(float xx){
        double x,y,tmp,ser;
        static double cof[6]={76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5};
        int j;

	    y=x=xx;
	    tmp=x+5.5;
	    tmp -= (x+0.5)*log(tmp);
	    ser=1.000000000190015;
	    for (j=0;j<=5;j++) ser += cof[j]/++y;
	    return -tmp+log(2.5066282746310005*ser/x);
    }

    int bnldev(float pp, int n){
        int j;
        static int nold=(-1);
        float am,em,g,angle,p,bnl,sq,t,y;
        static float pold=(-1.0),pc,plog,pclog,en,oldg;

	    p=(pp <= 0.5 ? pp : 1.0-pp);
	    am=n*p;
	    if (n < 25) {
		    bnl=0.0;
		    for (j=1;j<=n;j++)
    			if (ran1(idum) < p) ++bnl;
	    } else if (am < 1.0) {
		    g=exp(-am);
		    t=1.0;
		    for (j=0;j<=n;j++) {
			    t *= ran1(idum);
			    if (t < g) break;
		    }
		    bnl=(j <= n ? j : n);
	    } else {
		    if (n != nold) {
			    en=n;
			    oldg=gammln(en+1.0);
			    nold=n;
		    } if (p != pold) {
    			pc=1.0-p;
	    		plog=log(p);
		    	pclog=log(pc);
			    pold=p;
		    }
		    sq=sqrt(2.0*am*pc);
		    do {
    			do {
	    			angle=PI*ran1(idum);
		    		y=tan(angle);
			    	em=sq*y+am;
			    } while (em < 0.0 || em >= (en+1.0));
			    em=floor(em);
			    t=1.2*sq*(1.0+y*y)*exp(oldg-gammln(em+1.0)-gammln(en-em+1.0)+em*plog+(en-em)*pclog);
		    } while (ran1(idum) > t);
		    bnl=em;
	    }
	    if (p != pp) bnl=n-bnl;
	    return (int)bnl;
    }

    auto compute(unsigned i) -> bool {
		if (  flag[0] == 01 ){
			auto prng = __builtin_ipu_urand32()  ;
		}
		else if (  flag[0] == 11 ){
			auto prng = (float)__builtin_ipu_urand32() / (float)UINT_MAX ;  //[0,2^32-1]
		}
		else if (  flag[0] == 12 ){
        auto prng = (double)__builtin_ipu_urand64() / (double)ULLONG_MAX ;  //[0,2^64-1]
		}		
		else if (  flag[0] == 21 ){
            auto prng = gammln((float)__builtin_ipu_urand32() / (float)UINT_MAX) ;
		}
		else if (  flag[0] == 22 ){
            auto prng = bnldev((double)__builtin_ipu_urand64() / (double)ULLONG_MAX, 1 ) ;
		}
        return true;
    }
};

