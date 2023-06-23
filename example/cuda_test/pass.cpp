#include "kernel.h"

#include "pass.h"
#include "system/gui.h"
#include "cuda/data_view.h"

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
        
        m_stream->Synchronize();
    }
    m_timer.Stop();
}