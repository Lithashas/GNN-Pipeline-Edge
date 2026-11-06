# GNN-Pipeline-Edge
PIPELINED CPU–GPU EXECUTION OF GRAPH NEURAL NETWORK INFERENCE ON SHARED-MEMORY EDGE DEVICES

![HPC](https://img.shields.io/badge/Project-HPC--Research-blueviolet)
![C++](https://img.shields.io/badge/Language-C%2B%2B17-blue)
![CUDA](https://img.shields.io/badge/Hardware-NVIDIA%20CUDA-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Overview
This repository contains the implementation for my 3rd-year High-Performance Computing project: **"Pipelined CPU-GPU Execution of GNN Inference on Shared-Memory Edge Devices."**

The core objective is to eliminate the performance bottlenecks in Graph Neural Networks (GNNs) by separating **Irregular Sparse Aggregation** (CPU-bound) from **Dense Feature Transformation** (GPU-bound) and executing them in an asynchronous pipeline.

### The "Duality" Challenge
- **Aggregation Phase:** Messy "pointer-chasing" logic. Handled by **CPU (OpenMP)** to avoid GPU Warp Divergence.
- **Combination Phase:** Heavy GEMM math. Handled by **GPU (CUDA)** for maximum throughput.

---

## System Architecture: The Assembly Line
The project implements a **Multi-Stage Pipeline** to hide the latency of the sparse CPU work behind the fast dense GPU math.

```text
Time Step  | CPU Stage (Sparse Agg) | GPU Stage (Dense GEMM)
-----------|------------------------|-----------------------
    T1     |     Partition 1        |        (Idle)
    T2     |     Partition 2        |     Partition 1
    T3     |     Partition 3        |     Partition 2
    T4     |     (Barrier Sync)     |     Partition 3
```
## Target Datasets
I evaluate efficiency across varying graph scales using the **Open Graph Benchmark (OGB)**:
* **Cora / PubMed:** Development & Correctness verification.
* **ogbn-arxiv:** Primary performance benchmark.
* **ogbn-products:** High-throughput capacity stress-test.

---

## Author
**H. M. L. S. ABAYARATHNE**
* **Student ID:** 23/ENG/003

---
*This project is submitted as part of the 3rd Year High-Performance Computing curriculum.*
