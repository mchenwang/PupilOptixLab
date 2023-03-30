#include "post_process.cuh"
#include <device_launch_parameters.h>

// namespace Pupil {
// namespace cuda {
// CUDA_INLINE CUDA_DEVICE float3 ACESToneMapping(float3 color, float adapted_lum) noexcept {
//     const float A = 2.51f;
//     const float B = 0.03f;
//     const float C = 2.43f;
//     const float D = 0.59f;
//     const float E = 0.14f;

//     color *= adapted_lum;
//     return (color * (A * color + B)) / (color * (C * color + D) + E);
// }
// CUDA_INLINE CUDA_DEVICE float3 GammaCorrection(float3 color, float gamma) {
//     return make_float3(powf(color.x, 1.f / gamma), powf(color.y, 1.f / gamma), powf(color.z, 1.f / gamma));
// }

// CUDA_GLOBAL void ACESToneMapWithGammaCorrection(float4 *output_image, const float4 *input_image, uint2 size, float gamma) {
//     int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
//     int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
//     if (pixel_x >= size.x) return;
//     if (pixel_y >= size.y) return;
//     int pixel = pixel_x + size.x * pixel_y;
//     float3 color = make_float3(input_image[pixel]);
//     color = ACESToneMapping(color, 1.f);
//     color = GammaCorrection(color, gamma);
//     output_image[pixel] = make_float4(color, input_image[pixel].w);
// }

// CUDA_GLOBAL void ACESToneMapWithoutGammaCorrection(float4 *output_image, const float4 *input_image, uint2 size) {
//     int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
//     int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
//     if (pixel_x >= size.x) return;
//     if (pixel_y >= size.y) return;
//     int pixel = pixel_x + size.x * pixel_y;
//     float3 color = make_float3(input_image[pixel]);
//     color = ACESToneMapping(color, 1.f);
//     output_image[pixel] = make_float4(color, input_image[pixel].w);
// }

// CUDA_GLOBAL void OnlyGammaCorrection(float4 *output_image, const float4 *input_image, uint2 size, float gamma) {
//     int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
//     int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
//     if (pixel_x >= size.x) return;
//     if (pixel_y >= size.y) return;
//     int pixel = pixel_x + size.x * pixel_y;
//     float3 color = make_float3(input_image[pixel]);
//     color = GammaCorrection(color, gamma);
//     output_image[pixel] = make_float4(color, input_image[pixel].w);
// }

// void PostProcess(cudaStream_t stream, cudaEvent_t finished_event,
//                  float4 *output_image, const float4 *input_image,
//                  uint2 size, float gamma, unsigned int post_process_type) {
//     constexpr int block_size_x = 32;
//     constexpr int block_size_y = 32;
//     int grid_size_x = (size.x + block_size_x - 1) / block_size_x;
//     int grid_size_y = (size.y + block_size_y - 1) / block_size_y;
//     dim3 grid(grid_size_x, grid_size_y);
//     dim3 block(block_size_x, block_size_y);
//     if (post_process_type == 1) {
//         ACESToneMapWithGammaCorrection<<<grid, block, 0, stream>>>(
//             output_image, input_image, size, gamma);
//     } else if (post_process_type == 2) {
//         ACESToneMapWithoutGammaCorrection<<<dim3(grid_size_x, grid_size_y),
//                                             dim3(block_size_x, block_size_y),
//                                             0, stream>>>(
//             output_image, input_image, size);
//     }
//     /*switch (post_process_type) {
//         case 0:
//             break;
//         case 1: {
//             ACESToneMapWithGammaCorrection<<<grid, block, 0, stream>>>(
//                 output_image, input_image, size, gamma);

//         } break;
//         case 2: {
//             ACESToneMapWithoutGammaCorrection<<<dim3(grid_size_x, grid_size_y),
//                                                 dim3(block_size_x, block_size_y),
//                                                 0, stream>>>(
//                 output_image, input_image, size);
//         } break;
//         case 3: {
//             OnlyGammaCorrection<<<dim3(grid_size_x, grid_size_y),
//                                   dim3(block_size_x, block_size_y),
//                                   0, stream>>>(
//                 output_image, input_image, size, gamma);
//         } break;
//     }*/
//     cudaEventRecord(finished_event, stream);
// }
// }
// }// namespace Pupil::cuda

#include <iostream>
__global__ void my_kernel() { /* ... */
}

void PostProcessXX() {
    my_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    std::cout << "Kernel launched!" << std::endl;
}