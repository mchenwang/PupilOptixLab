#include "cuda/kernel.h"
#include "kernel.h"
#include "cuda/vec_math.h"

void SetColor(uint2 size, unsigned int m_frame_cnt, Pupil::cuda::RWArrayView<float4> &m_output, Pupil::cuda::Stream *stream) {
    Pupil::cuda::LaunchKernel2D(
        size, [=] __device__(uint2 pixel_id, uint2 size) {
            float3 color = 0.5f * make_float3(
                                      cos(((float)pixel_id.x) / size.x + m_frame_cnt / 100.f),
                                      sin(((float)pixel_id.y) / size.y + m_frame_cnt / 100.f),
                                      0.7f) +
                           0.5f;
            m_output[pixel_id.x + pixel_id.y * size.x] = make_float4(color, 1.f);
        },
        stream);
}