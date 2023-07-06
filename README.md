# PupilOptixLab

PupilOptixLab is a lightweight real-time ray tracing framework based on OptiX7 which is designed for rapid implementation of ray tracing algorithms on GPU.

A basic path tracer has been implemented in [`example/path_tracer`](https://github.com/mchenwang/PupilOptixLab/tree/main/example/path_tracer).

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/PupilOptixLab.jpg)

## Features

- mistuba3-style scenes format
- pt with mis
- OptiX7 denoiser
- material: diffuse, conductor, rough conductor, dielectric, rough dielectric, plastic, rough plastic
- asynchronous GUI thread
- support native cuda code
- [wavefront path tracer](https://github.com/mchenwang/WavefrontPathTracer): about 3x performance improvement

## Prerequisites

- CMake 3.25.2+ (to support `-std=c++20` option for nvcc)
- Visual Studio 2022
- NVIDIA graphics card with OptiX7 support
- CUDA 12.0+ (tested on 12.1)
- OptiX 7.5 (for the built-in sphere intersection and higher versions are currently not supported due to APIs changes)

## Rendering Example

### classroom

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/classroom.png)

### bathroom

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/bathroom1.png)

### bathroom2

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/bathroom2.png)

### bedroom

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/bedroom.png)

### kitchen

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/kitchen.png)

### living-room-2

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/livingroom2.png)

### living-room-3

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/livingroom3.png)

### staircase

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/staircase.png)

### staircase2

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/staircase2.png)

### veach-ajar

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/veach-ajar.png)

### veach-mis

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/veach-mis.png)

### lamp

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/lamp.png)
