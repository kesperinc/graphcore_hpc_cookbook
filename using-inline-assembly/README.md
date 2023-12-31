# Using inline assembly
This recipe shows you how you might write some IPU assembly inline in a C++ vertex, by using `popc`'s support for the
extended assembly (`__asm`) syntax.

Before you start writing IPU assembly, make sure you've read the [Poplar Vertex Assembly Programming Guide](https://docs.graphcore.ai/projects/assembly-programming/en/latest/hints_tips.html),
and have access to the Tile Worker ISA (which explains
the instructions that are availbale.) 

We can't give you access to the ISA docs -- please ask your Graphcore support engineer
for access instead. You can also trawl the [Poplibs source code on GitHub](https://github.com/graphcore/poplibs) for ideas.


## Why would you do this?!
Sometimes you need to use some of the IPU's special instructions that haven't been exposed in an elegant way yet in the SDK.
For example, in this example we'll show how to generate random numbers using the special hardware instruction `urand32`.

Before you do this, though, consider whether the well-tested SDK already supports your needs (e.g. in our case, 
maybe the `poprand` library already has what we need, and could be used more idiomatically).

Every time you write assembly, your code gets less portable and more difficult to test, so use this only as a last resort!

## First check if there's some instrinsic function that already does what you need
Even if it's not immediately apparent in the API documentation, the `popc` compiler also
comes with some [intrinsic functions](https://en.wikipedia.org/wiki/Intrinsic_function) that might be exposing the functionality you need.

You can inspect the intrinsics by looking at the strings in the `libpoplar.so` library. For example, in SDK v1.4
```sh
> strings libpoplar.so | grep '__builtin_ipu' | sort -u 

__builtin_ipu_f32atan
__builtin_ipu_f32class
__builtin_ipu_get
__builtin_ipu_get_scount_l
__builtin_ipu_get_scount_u
__builtin_ipu_get_tile_id
__builtin_ipu_get_vertex_base
__builtin_ipu_is_worker_mode
__builtin_ipu_put
__builtin_ipu_tapack
__builtin_ipu_uget
__builtin_ipu_uput
__builtin_ipu_urand32
__builtin_ipu_urand64
```

We see that there is already an intrinsic function for the code we're about to write (invoking the `urand32` instruction).
(_Note: in SDK v1.4 the `__builtin_ipu_urand32()` instrinsic actually returns an incorrectly scaled `float` - it should be using
a subsequent `f32sufromui` operation to convert correctly to `float`. This bug is confirmed as fixed in SDK v2+_).


### Using inline IPU assembly in a C++ vertex
So now that the "don't do this unless you need to" warnings are out of the way, let's write some inline assembly.
This only makes sense from within Custom Vertexes (i.e. you wouldn't be doing this in your host C++ program).

Say we have a basic C++ vertex:
```C++
class InlineAssemblyVertex: public Vertex {

public:
    bool compute() {
        ...
#if defined(__IPU__) && defined(__POPC__)
        // Let's do some inline assembly here
#else
        // Maybe a portable C++ version or dummy values here?
#endif        
        ...
    }    
};
```

Note that we've protected sections of IPU assembly by checking whether `__IPU__` is defined, which allows us to keep code portable 
(e.g. be able to run your codelets on the `IPUModel`, or differentiate between `ipu1` and `ipu2` target code).

`popc` is LLVM-based, so we can use the LLVM and GCC extended assembly (`__asm`) syntax. Information on the extended assembly syntax
is thin on the ground, but there's a pretty good write-up 
[here](https://www.felixcloutier.com/documents/gcc-asm.html). Most examples on the Internet are for
x86, and it can be confusing to try and write this for the IPU instead.

For the IPU, it's important to remember that 
* IPU assembly separates operands with commas, not spaces
* register names start with a `$`, e.g. `$m0`. Don't use `%` syntax like x86 `%eax`.
* Use the `%[name]` syntax for variable names
* Remember to mark the registers you use as clobbered.
* Remember that some instructions only work with memory register operands (e.g. `$m0`), while others work with
  arithmetic register operands (e.g. `$a0`). You can use the `atom` instruction to move a result from one kind of register to the other.
* Unlike some x86 instructions, operands are either literals or all in registers; you don't have memory location operands.
  So you need to use the load/store instructions (e.g. `ld32`) explicitly if you want to write something back to memory
  from assembly. You can also capture outputs to C++ variables (or assembly inputs from C++ variables).
* You can use multiple lines if you separate them with the `"\n\t"` syntax  

   
So let's take a look at a real example: generating a random uniform number that is a `float`. 
1. We'll use the `urand32` instruction to generate a uniform random number that is a 32-bit `unsigned int`, and 
store the output in `$a0`.
2. We'll convert the 32-bit `unsigned int` to a `float` using the `f32sufromui` instruction, and store that in the assembly variable `%[result]`
4. We'll tell the compiler that the contents of that `%[result]` memory register should go in the `tmp` C++ variable
5. We'll let the compiler know that the `$a0` register has been clobbered

Note that if you actually want to generate random numbers, you should set up the PRNG state registers appropriately, and using the
`poprand` library functions is probably the easiest way. You'd also want to modify the PRNG state
to avoid generating the same number over and over. We don't show that here, but the [poprand sources](https://github.com/graphcore/poplibs/blob/2bc6b6f3d40863c928b935b5da88f40ddd77078e/lib/poprand/codelets/asm/Seeds.S)
show how this can be done.

Here's the final inline assembly code:

```C++
class InlineAssemblyVertex: public Vertex {

public:
    bool compute() {
        float tmp = 0.;
        ...
#if defined(__IPU__) && defined(__POPC__)
        __asm(
                "urand32 $a0\n\t"
                "f32sufromui %[result], a0\n\t"
        :[result]"=r"(tmp)
        :
        :"$a0");
#else
        // Maybe a portable C++ version or dummy values here?
#endif        
        ...
    }    
};
```

Good luck, and remember to use this sparingly.


