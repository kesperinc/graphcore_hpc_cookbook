Table of Contents
=================

* [Monte Carlo method implementation examples](#monte-carlo-method-implementation-examples)
* [Usage](#usage)
* [PI Implementation](#pi-implementation)
   * [MultiVertex implementation (vertex_ipu_pi)](#multivertex-implementation-vertex_ipu_pi)
   * [Poplibs implementatio (poplibs_ipu_pi)](#poplibs-implementatio-poplibs_ipu_pi)
   * [Iterative implementation (iterative_ipu_pi)](#iterative-implementation-iterative_ipu_pi)
* [Integrals implementation](#integrals-implementation)
   * [MultiVertex implementation (vertex_ipu_*)](#multivertex-implementation-vertex_ipu_)

# Monte Carlo method implementation examples

Example programs to estimate PI and definite integral value using Monte Carlo method
for IPU architecture.

# Usage

All example programs can take up to 4 parameters:
```
--iterations <number> (default 30.000.000) defines number of <chunk_size> check loops
--chunk_size <number> (default 10.000.000) defines size of the buffer for samples to be checked
--num_ipus  <number>  (default 1)          defines how many IPUs will be used for computation
--precision <number>  (default 10)         defines number of digits of the result number to be printed
```

# PI Implementation

Every implementation generates `--iterations` number of points with random coordinates `x` and `y`.
Every point is checked against circle equation `x^2 + y^2 <= 1` and, when it is fulfilled, counter is incremented.
Result is sum of samples matching equation multiplied by 4 and divided by number of samples.

## MultiVertex implementation (vertex_ipu_pi)

This implementations is codelet based. It creates output tensor with `num_ipus * tiles * threads` elements.
Codelet code is available in [pi/pi_vertex.cpp](pi/pi_vertex.cpp)
Sample data is also generated on IPU with `__builtin_ipu_urand32()` call.
Result vector is sent to host where PI is estimated based on that.

## Poplibs implementatio (poplibs_ipu_pi)

This implementation shows how to use PopLibs operators. No codelets used.
Random data is generated on IPU using `poprand::uniform()`. Comparison is done with
`popops::expr::Lte()` and square is `popops::expr::Square`.
Result vector is sent to host where PI is estimated based on that.

## Iterative implementation (iterative_ipu_pi)

This implementation is again PopLibs based, however, input data is generated on the host.

# Integrals implementation

## MultiVertex implementation (vertex_ipu_*)

All implementations are codelet based and differ only with compute vertex code.
Codelets are available in [integrals/integrals_vertex.cpp](integrals/integrals_vertex.cpp)
You can read more about those integrals here:
https://www.researchgate.net/publication/320738917_Vectorized_algorithm_for_multidimensional_Monte_Carlo_integration_on_modern_GPU_CPU_and_MIC_architectures

