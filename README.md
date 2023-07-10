# PupilOptixLab

PupilOptixLab is a lightweight real-time ray tracing framework based on OptiX7 which is designed for rapid implementation of ray tracing algorithms on GPU.

Some ray tracing methods will be implemented in [PupilRay](https://github.com/mchenwang/PupilRay).

## Features

- mistuba3-style scenes format
- pt with mis
- OptiX7 denoiser
- material: diffuse, conductor, rough conductor, dielectric, rough dielectric, plastic, rough plastic
- asynchronous GUI thread
- support native cuda code
- camera interaction (mouse dragging \ keyboard moving with WASDQR)
- [wavefront path tracer](https://github.com/mchenwang/WavefrontPathTracer): about 3x performance improvement

## Prerequisites

- CMake 3.25.2+ (to support `-std=c++20` option for nvcc)
- Visual Studio 2022
- NVIDIA graphics card with OptiX7 support
- CUDA 12.0+ (tested on 12.1)
- OptiX 7.5 (for the built-in sphere intersection and higher versions are currently not supported due to APIs changes)

## screenshot

![](https://github.com/mchenwang/PupilOptixLab/raw/main/image/PupilOptixLab.jpg)

