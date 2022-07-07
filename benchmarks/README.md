Table of Contents
=================

# Benchmark examples

## Peak matMul FLOPS benchmark

Simple benchmark measuring FLOPS for matrix multiplication on the IPU.

The flow is to generate random matrices using poprand[https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs_api.html#random-number-operations-poprand] and do grouped matrix multiplication[https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs/poplin/MatMul.html?highlight=matMulGrouped#_CPPv4N6poplin13matMulGroupedERN6poplar5GraphERKN6poplar6TensorERKN6poplar6TensorERN6poplar7program8SequenceERKN6poplar4TypeERKN6poplar12DebugContextERKN6poplar11OptionFlagsEPN6matmul13PlanningCacheE].

## Building Prerequisites



## Usage

All usage options can be obtained by using `--help`

```
  --help                         Help message
  -s [ --size ] arg (=1024)      Size of matrix
  -i [ --iterations ] arg (=10)  Iterations
  -g [ --groups ] arg (=1)       Groups
  -t [ --type ] arg (=float)     Type (half/float)
  --partials_type arg (=float)   Partial sum type (half/float)
  --memory_proportion arg (=0.6) Available memory proportion (0.01 - 0.99)
  --fast_reduce                  Enable fast reduce
  --multi_stage_reduce           Enable Multi stage reduce
  --remap_output_tensor          Remap output tensor
  --128bit_load                  128-bit load
```

Flags are matching options for matMul operation in Poplar(here[https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs/poplin/MatMul.html?highlight=matMul#_CPPv4N6poplin6matMulERN6poplar5GraphERKN6poplar6TensorERKN6poplar6TensorERN6poplar7program8SequenceERKN6poplar4TypeERKN6poplar12DebugContextERKN6poplar11OptionFlagsEPN6matmul13PlanningCacheE]).
