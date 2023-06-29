#include "kernel.h"

#include "pass.h"
#include "system/gui.h"
#include "cuda/data_view.h"
#include "cuda/kernel.h"

using namespace Pupil;

unsigned int m_frame_cnt = 0;
cuda::RWArrayView<float4> m_output;

CudaPass::CudaPass(std::string_view name) noexcept
    : Pass(name) {
    m_stream = std::make_unique<cuda::Stream>();
}

void CudaPass::Run() noexcept {
    m_timer.Start();
    {
        m_frame_cnt++;
        auto &frame_buffer =
            util::Singleton<GuiPass>::instance()->GetCurrentRenderOutputBuffer().shared_buffer;

        m_output.SetData(frame_buffer.cuda_ptr, 512 * 512);

        SetColor(m_frame_cnt, m_output, m_stream.get());
        // FIXME: To be mixed compiling
        // Pupil::cuda::LaunchKernel2D(
        //     uint2{ 512, 512 }, [out = m_output, f = m_frame_cnt] __device__(uint2 pixel_id, uint2 size) {
        //         float3 color = 0.5f * make_float3(
        //                                   cos(((float)pixel_id.x) / size.x + f / 100.f),
        //                                   sin(((float)pixel_id.y) / size.y + f / 100.f),
        //                                   0.7f) +
        //                        0.5f;
        //         out[pixel_id.x + pixel_id.y * size.x] = make_float4(color, 1.f);
        //     },
        //     m_stream.get());

        m_stream->Synchronize();
    }
    m_timer.Stop();
}