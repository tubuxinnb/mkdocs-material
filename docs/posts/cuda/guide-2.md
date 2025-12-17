---
date: 
  created: 2025-12-07
categories:
  - NOTE 
  - CUDA
links:
  - posts/cuda/guide-1.md
  - posts/cuda/guide-3.md
draft: true
---

# CUDA Programming Guide-2
Now it comes to the second part of the guide, which is about programming GPUs in CUDA and introduces some basic concepts in the CUDA programming model.

<!-- more -->

## 2.1. Intro to CUDA C++
1. Focus on CUDA runtime API.
> [CUDA Runtime API and CUDA Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#cuda-platform-driver-and-runtime) discusses the difference between the APIs and CUDA driver API discusses writing code that mixes the APIs.
2. [The CUDA Quickstart](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) Guide for basic installation.

### 2.1.1. Compilation with NVCC
NVCC is the CUDA compiler.
> The nvcc chapter of this guide covers common use cases of nvcc, and complete documentation is provided by the [nvcc user manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
### 2.1.2. Kernels
kernels are functions executed by the GPU and launched by the CPU.
#### 2.1.2.1. Specifying Kernels
`__global__` is a keyword that specifies a function as a kernel.
```
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{
...
}
```
#### 2.1.2.2. Launching Kernels
The num of threads executing the kernel is specified as part of the execution configuration.

2 ways to launch kernels:
1. **triple chevron notation**
2. `cudaLaunchKernelEx`, which will be talked later.

##### 2.1.2.2.1 Triple Chevron Notation

Now there are some concrete grammars and definitions, referred to the source of guide. Here list some keywords: 
`dim3`: CUDA type for description of grid and block

`<<dim3 grid, dim3 block>>`

`threadIdx` gives the index of a thread within its thread block. Each thread in a thread block will have a different index.

`blockDim` gives the dimensions of the thread block, which was specified in the execution configuration of the kernel launch.

`blockIdx` gives the index of a thread block within the grid. Each thread block will have a different index.

`gridDim` gives the dimensions of the grid, which was specified in the execution configuration when the kernel was launched.

`cuda::ceil_div` does the ceiling divide to calculate the number of blocks needed for a kernel launch.
```
int blocks = cuda::ceil_div(vectorLength, threads);
```

### 2.1.3. Memory in GPU Computing
A, B, C in `vecadd` should be accessible for threads. The ways are various. Here are 2, more in the [Unified Memory]().

#### 2.1.3.1 Unified Memory
Usage: Memory is allocated using the `cudaMallocManaged` API or by declaring a variable with the `__managed__` specifier. 
Function: the allocated memory is accessible to the GPU or CPU whenever either tries to access it.
Unified memory can be released using cudaFree.

!!! quote

    On some Linux systems, (e.g. those with address translation services or heterogeneous memory management) all system memory is automatically unified memory, and there is no need to use cudaMallocManaged or the __managed__ specifier.

#### 2.1.3.2 Explicit Memory Management
Usage: Memory is allocated using the `cudaMalloc` API, and data is copied using the `cudaMemcpy` API.
other used APIs:
`cudaDeviceSynchronize()`: synchronize the host and device, almost the host code will be blocked until the device code finishes.
`cudaMemset()`: set device memory to a value. 

**Advantages**: 
1. more control over memory allocation and data movement.
2. typically faster than unified memory.

However, explicite memory managemen is more complex and requires more care.

### 2.1.4. Synchronizing CPU & GPU
`cudaDeviceSynchronize`, which blocks the host thread until all previously issued work on the GPU has completed.

!!! quote 

    In larger applications, there may be multiple streams executing work on the GPU and cudaDeviceSynchronize will wait for work in all streams to complete. In these applications, using Stream Synchronization APIs to synchronize only with a specific stream or CUDA Events is recommended. These will be covered in detail in the Asynchronous Execution chapter.

### 2.1.5. Putting all together
Here the guide provides 2 complete [examples](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#putting-it-all-together), and introduces `__syncthreads()`
which is used to synchronize threads within a thread block.
the synchronization between blocks is provided by [Coopertative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#cooperative-groups).

[CUDA synchronization primitives](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#cooperative-groups)

### 2.1.6. Runtime INitialization

CUDA runtime creates a [CUDA context](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#driver-api-context) for each GPU device.
As of CUDA 12.0, the execution of runtime initialization is provided by `cudaInitDevice` and `cudaSetDevice` and destroyed by `cudaDeviceReset`.

### 2.1.7. Error Checking in CUDA
A utility macro follows:
```
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)
```
the `cudaGetErrorString` in the macro returns a human readable string describing the meaning of a specific `cudaError_t` value. 

### 2.1.8. Device & Host Functions

use the `__device__` and `__host__` keywords to specify whether a function is executed on the GPU or the CPU.

### 2.1.9. Variable Specifiers

- `__device__` specifies that a variable is stored in [Global Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-global-memory)
- `__constant__` specifies that a variable is stored in [Constant Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-constant-memory)
- `__managed__` specifies that a variable is stored as [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-memory)
- `__shared__` specifies that a variable is store in [Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory)

!!! quote "Both specifiers in the same function"
    When a function is specified with `__host__ __device__`, the compiler is instructed to generate both a GPU and a CPU code for this function. In such functions, it may be desirable to use the preprocessor to specify code only for the GPU or the CPU copy of the function. Checking whether `__CUDA_ARCH_ `is defined is the most common way of doing this, as illustrated in the example below.

### 2.1.10. Thread Block Clusters

Thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU.

The size of a cluster is user-defined, the maximum size is 8 blocks.
`cudaOccupancyMaxPotentialClusterSize` returns the maximum number (mabye > or < 8) of blocks that can be launched in a cluster.

thread blocks in a cluster can synchronize with each other using `cluster.sync()` in [cooperative groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#cooperative-groups), and have access to distributed shared memory.
Distributed shared memory is the combined shared memory of all thread blocks in the cluster. 

An Example: 
```
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // IMPORTANT: The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

## 2.2. Writing CUDA SIMT Kernels
### 2.2.1. Basics of SIMT
### 2.2.2. Thread Hierarchy
- `gridDim.[x|y|z]`: Size of the grid in the x, y and z dimension respectively. These values are set at kernel launch.

- `blockDim.[x|y|z]`: Size of the block in the x, y and z dimension respectively. These values are set at kernel launch.

- `blockIdx.[x|y|z]`: Index of the block in the x, y and z dimension respectively. These values change depending on which block is executing.

- `threadIdx.[x|y|z]`: Index of the thread in the x, y and z dimension respectively. These values change depending on which thread is executing.

### 2.2.3. GPU Device Memory Spaces
| Memory Type | Scope | Lifetime | Location |
| --- | --- | --- | --- |
| Global | Grid | Application | Device |
| Constant | Grid | Application | Device |
| Shared | Block | Kernel | SM |
| Local | Thread | Kernel | Device |
| Register | Thread | Kernel | SM |

#### 2.2.3.1. Global Memory

Prior to a kernel launch: Global memory is allocated and initialized by CUDA APIs.

During kernel execution: Global memory is accessed (read&write) by CUDA APIs.

A kernel compltetes execution: the data in global memory can be written back to host memory or used by another kernels.

!!! quote "data copy"

    Since CUDA kernels launched from the host have the return type `void`, the only way for numerical results computed by a kernel to be returned to the host is by writing those results to global memory.

#### 2.2.3.2. Shared Memory

Phyiscal: shared memory and data L1 cache are the same, located inner a SM.
Scope: all threads in a block.
Lifetime: the whole kernel execution.
others: use `__syncthreads()` for synchronization of threads in a block.

#### 2.2.3.3. Registers

Physical: registers are located on the SM.
Scope: for single thread's local storage.
Lifetime: thread's execution

#### 2.2.3.4. Local Memory
Physical: local memory is located on Device Memory.
Scope: for single thread's local storage.
Lifetime: lthread's execution.
others: [Coalesced Global Memory Access](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-coalesced-global-memory-access).

#### 2.2.3.5. Constant Memory
Physical: fixed 64KB typically.
Scope: all threads in a grid.
Lifetime: the application's lifetime.
others: Constant memory is accessible from all the threads within the grid and from the host through the runtime library (`cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`).

#### 2.2.3.6. Caches

L2 Cache is shared among all the SMs, while L1 Cache is shared among all the threads within a SM along with shared memory.

Caches' behaviors are configurable:

!!! quote 
    
    The L2 and L1 caches can be controlled via functions that allow the developer to specify various caching behaviors. The details of these functions are found in [Configuring L1/Shared Memory Balance](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernel-l1-shared-config), [L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#advanced-kernels-l2-control), and [Low-Level Load and Store Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#low-level-load-store-functions)

#### 2.2.3.7. Texture and Surface Memory

No performance advantages now.

#### 2.2.3.8. Distributed Shared Memory

Accessing data in distributed shared memory requires all the thread blocks to exist.
This may be a strict condition, using `cluster.sync()` to ensure.
#### 2.2.4. and So on (later update)
#### 2.3. Asynchronous Execution

Asynchronous execution is a way to improve performance. Now skip first.

## 2.4. Unified and System Memory
Unified memory has several different manifestations, which depend upon the OS, driver version and GPU.

!!! note Unified Memory

    - Limited Unified Memory - A unified memory paradigm with some limitations
    - Full Unified Memory - Full support for unified memory features
    - Full Unified Memory with Hardware Coherency - Full support for unified memory using hardware capabilities
    - Unified memory hints - APIs to guide unified memory behavior for specific allocations

!!! note Page-locked Host Memory

    - [Mapped memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-mapped-memory) - A mechanism (different from unified memory) for accessing host memory directly from a kernel

Heterogeneous Managed Memory (HMM) in Linux enables software coherency for full unified memory.

Address Translation Services provides hardware coherency for full unified memory.

### 2.4.1. Unified Virtual Address Space

### 2.4.2. Unified Memory
!!! quote 

    On systems with HMM or ATS, all system memory is implicitly managed memory, regardless of how it is allocated. No special allocation is needed.

#### 2.4.2.1. Unified Memory Paradigms
4 paradigms:
- [Full support for explicit managed memory allocations](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-memory-full)
- [Full support for all allocations with software coherence](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-memory-full)
- [Full support for all allocations with hardware coherence](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-address-translation-services)
- [Limited unified memory support](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-limited-unified-memory-support)

When full support is available, it can either require explicit allocations, or all system memory may implicitly be unified memory. When all memory is implicitly unified, the coherence mechanism can either be software or hardware. Windows and some Tegra devices have limited support for unified memory.

![](image.png)
All current GPUs use a unified virtual address space and have unified memory available. When cudaDevAttrConcurrentManagedAccess is 1, full unified memory support is available, otherwise only limited support is available. When full support is available, if cudaDevAttrPageableMemoryAccess is also 1, then all system memory is unified memory. Otherwise, only memory allocated with CUDA APIs (such as cudaMallocManaged) is unified memory. When all system memory is unified, cudaDevAttrPageableMemoryAccessUsesHostPageTables indicates whether coherence is provided by hardware (when value is 1) or software (when value is 0).