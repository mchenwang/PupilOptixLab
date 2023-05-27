#include "test.h"
#include "cuda/vec_math.h"

__global__ void Test(float4 *output_image, uint2 size, unsigned int frame_cnt) {
    int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixel_x >= size.x) return;
    if (pixel_y >= size.y) return;
    int pixel_index = pixel_x + size.x * pixel_y;
    float3 color = 0.5f + 0.5f * make_float3(
                                     cos(((float)pixel_x) / size.x + frame_cnt / 100.f),
                                     sin(((float)pixel_y) / size.y + frame_cnt / 100.f),
                                     0.7f);
    output_image[pixel_index] = make_float4(color, 1.f);
}

void CudaSetColor(cudaStream_t stream, Pupil::cuda::RWArrayView<float4> &output_image, uint2 size, unsigned int frame_cnt) {
    constexpr int block_size_x = 32;
    constexpr int block_size_y = 32;
    int grid_size_x = (size.x + block_size_x - 1) / block_size_x;
    int grid_size_y = (size.y + block_size_y - 1) / block_size_y;
    Test<<<dim3(grid_size_x, grid_size_y),
           dim3(block_size_x, block_size_y),
           0, stream>>>(output_image.GetDataPtr(), size, frame_cnt);
}
