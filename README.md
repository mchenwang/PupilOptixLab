# PupilOptixLab

PupilOptixLab is a lightweight real-time ray tracing framework based on OptiX7. A basic path tracer has been implemented on PupilOptixLab.

![](.\image\PupilOptixLab.jpg)

## Features

- mistuba3-style scenes format

- pt with mis
- OptiX7 denoiser
- material: diffuse, conductor, rough conductor, dielectric, rough dielectric, plastic, rough plastic
- asynchronous GUI thread
- support native cuda code

## Prerequisites

- Visual Studio 2022
- NVIDIA graphics card with OptiX7 support
- CUDA 11.6+ (tested on 11.6 and 12.1)
- OptiX 7.5+ (for built-in sphere intersection)

## Rendering Example

### bathroom

![](./image/bathroom1.png)

### bathroom2

![](./image/bathroom2.png)

### bedroom

![](./image/bedroom.png)

### kitchen

![](./image/kitchen.png)

### living-room-2

![](./image/livingroom2.png)

### living-room-3

![](./image/livingroom3.png)

### staircase

![](./image/staircase.png)

### staircase2

![](./image/staircase2.png)

### veach-ajar

![](./image/veach-ajar.png)

### veach-mis

![](./image/veach-mis.png)

### lamp

![](./image/lamp.png)