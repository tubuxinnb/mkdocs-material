---
date: 
  created: 2025-12-07
categories:
  - NOTE
  - CUDA
links:
  - posts/cuda/guide-2.md
  - posts/cuda/guide-3.md
  - posts/cuda/guide-4.md
---

# CUDA Programming Guide-1

!!! abstract "导言"

    CUDA is a parallel computing platform and programming model that enables dramatic increases in computing performance by harnessing the power of the graphics processing unit (GPU). Wishing to learn more about CUDA, this guide is a great place to start.





[Source link: CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

<!-- more -->

The guide is broke down into 5 primary parts: 

Part 1: Introduction and Programming Model Abstract 

Part 2: [Programming GPUs in CUDA](cuda/guide-2.md)

Part 3: Advanced CUDA

Part 4: CUDA Features

Part 5: Technical Appendices

Parts 1-3 provide a guided learning experience for developers new to CUDA, though they also provide insight and updated information useful for CUDA developers of any experience level.

Parts 4 and 5 provide a wealth of information about specific features and detailed topics, and are intended to provide a curated, well-organized reference for developers neeing to know more details as they write CUDA applications.

## Part 1: Introduction and Programming Model Abstract

### 1.1 Intro

A GPU provides much higher **instruction throughput and memory bandwidth** than a CPU within a similar price and power envelope.

Other computing devices, like FPGAs, are also very energy efficient, but offer much less **programming flexibility** than GPUs.

![gpu-devotes-more-transistors-to-data-processing](https://picgo-picbed-buxin.oss-cn-hangzhou.aliyuncs.com/picgo/gpu-devotes-more-transistors-to-data-processing.png)

Specialized library: cuBLAS, cuFFT, cuDNN, CUTLASS

higher-level language: Triton, tilelang ...

The [NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub) contains resources, examples, and tutorials to teach GPU and CUDA computing.

### 1.2 Programming Model

#### 1.2.1. Heterogeneous System

Device code, kernel, kernel launch

#### 1.2.2. GPU HW Model

GPU = A collection of Streaming Multiprocessors (SMs)

GPC = Graphics Processing Clusters = a group of SMs

SM = 1 local Reg File + 1 Uni-Data Cache + multi Function Units

Uni-Data Cache = shared memory + L1C, software configurable

<img src="https://picgo-picbed-buxin.oss-cn-hangzhou.aliyuncs.com/picgo/gpu-cpu-system-diagram-20251207153451346.png" alt="The CUDA programming model view of CPU and GPU components and connection" style="zoom:25%;" />

##### 1.2.2.1 Thread block & grid

block: a set of threads

grid: block organization, the blocks are the same size.

dimension: blocks and grids include the concept of dimension.

in kernel launch, a execution configuration specifies the grid an block dimensions, also the SM, stream and Cluster related params.

every thread has own id, determined by the block location, grid location. And the thread could know the size of blocks and grids.

**A block of threads are running in a single SM.**

**No guarantee of scheduling between thread blocks, so no data dependency between thread blocks**

![Thread blocks scheduled on SMs](https://picgo-picbed-buxin.oss-cn-hangzhou.aliyuncs.com/picgo/thread-block-scheduling.png)

###### Thread Block Clusters

a group of thread blocks, organized by **adjacent** blocks in a grid, offers some synchronization and communications

Specifically, A cluster is executed in a single GPC

Threads in different block but in the same cluster can sync & comm by Cooperative Groups

Now assume T is Thread, C is Cluster, B is block, G is Grid, 

Ts in a C but different B can communicate and synchronize by interfaces provided by Cooperative Groups and access their shared memory -- distributed shared memory.

cluster size is determined by HW of GPU.

###### Warp & SIMT

In a B, Ts are organized into groups of 32 Ts called Warp. Now W is Warp.

Ts in W execute the same codes by Single-Instruction Multiple-Threads (SIMT)

T has a W id, 0-31. 

The way that Ts into W is **predictable** (?) cited [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-hardware-multithreading).

Control flow branch may causes some Ts in W masked off, as the fig follows.

![Warp lanes are masked off when not active%](https://picgo-picbed-buxin.oss-cn-hangzhou.aliyuncs.com/picgo/active-warp-lanes.png)

CUDA cannot see Warp.

helpful for  [global memory coalescing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-coalesced-global-memory-access) and [shared memory bank access patterns](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory-access-patterns).

The num of Ts in a B better are a multiple of 32 (Ts size in a W).

#### 1.2.3 GPU Memory 

##### 1.2.3.1 DRAM Memory in Heterogeneous Systems

System Memory = Host Memory = CPU DRAM

GPU DRAM = Global Memory, accessible to all SMs.

Virtual Memory in GPU: can recognize which GPU, whether in GPU memory or CPU memory

There are lots of API for memory control.

CUDA can use Unified Memory -- a mechanism, to automatically handling the placement of memory in the runtime.

##### 1.2.3.2 GPU on-chip Memory

Reg files (for temp variables) + shared memory (can be used for data exchange in Block & Cluster)

on-chip Memory is shared by all Ts in a B.

However, the Reg files is allocated to Ts while the Shared memory is allocated to Bs.

###### **Caches**

 L1 is from shared memory, L2 is for all SMs in GPU, **constant cache** is for constant variables in global memory.

##### 1.2.3.3 Unified Memory

> Section [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#memory-unified-memory) introduces the different categories of unified memory systems. Section [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#um-details-intro) contains many more details about use and behavior of unified memory in all situations.

CPU codes only access CPU memory while GPU kernels only access GPU memory in addition to mapped memory.

CUDA APIs work for the data copies between them.

### 1.3 The CUDA Platform

#### 1.3.1. Compute Capability (CC)  & SM Versions

GPU has a CC number documented in the  [Section 5.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities) appendix, looking like X.Y

E.g: CC 12.0 - sm_120

#### 1.3.2. CUDA Toolkit & NV Driver

Nvidia Driver = GPU OS

CUDA Toolkit = Lib + Headers + Tools

CUDA runtime = one Library of CUDA Toolkit

> About **Compatibility**: The [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) document provides full details of compatibility between different GPUs, NVIDIA Drivers, and CUDA Toolkit versions.

Driver API covers the CUDA Runtime API

> The full API reference for the CUDA runtime API functions can be found in the [CUDA Runtime API Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html) .

#### 1.3.3. Parallel Thread Execution (PTX)

PTX is an invisible virtual instruction set architecture, which aligns with SM Versions.

> Full documentation on PTX can be found in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) .

#### 1.3.4 Cubins & Fatbins

Cubin: CUDA binary for physical GPU.

Fatbin: a container of mutiple versions of cubins & PTX

![Fatbin containers within executables or libraries can contain multiple GPU code versions](https://picgo-picbed-buxin.oss-cn-hangzhou.aliyuncs.com/picgo/fatbin.png)

##### 1.3.4.1. Binary Compatibility

1. a cubin's version is CC8.6, then GPU with 8.x where x >= 6 can execute while GPU with 8.x where x < 6 cannot.

2. Where X.Y, different X is not coampatible.

##### 1.3.4.2. PTX Compatibility

In fatbins, if PTX is compute_80, it can be JIT compiled at application runtime for any CC equal or higher to the CC of the PTX code (such as sm_120).

##### 1.3.4.3. Just-in-Time Compilation

NVRTC can be used to compile CUDA C++ device code to PTX at runtime.
