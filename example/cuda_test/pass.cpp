#include "pass.h"
#include "test.h"
#include "system/gui.h"
#include "cuda/data_view.h"

using namespace Pupil;

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

        cuda::RWArrayView<float4> output;
        output.SetData(frame_buffer.cuda_ptr, 512 * 512);
        CudaSetColor(m_stream->GetStream(), output, make_uint2(512, 512), m_frame_cnt);
        m_stream->Synchronize();
    }
    m_timer.Stop();
}