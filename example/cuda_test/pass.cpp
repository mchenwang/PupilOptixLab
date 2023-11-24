#include "kernel.h"
#include "pass.h"

#include "system/buffer.h"
#include "system/event.h"
#include "system/gui/pass.h"

#include "cuda/util.h"
#include "cuda/kernel.h"

using namespace Pupil;

CudaPass::CudaPass(std::string_view name) noexcept
    : Pass(name) {
    m_stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::Render);

    auto       buf_mngr = util::Singleton<BufferManager>::instance();
    BufferDesc default_frame_buffer_desc{
        .name           = "default buffer",
        .flag           = EBufferFlag::AllowDisplay,
        .width          = static_cast<uint32_t>(512),
        .height         = static_cast<uint32_t>(512),
        .stride_in_byte = sizeof(float) * 4};
    buf_mngr->AllocBuffer(default_frame_buffer_desc);

    BufferDesc test_buffer_desc{
        .name           = "test buffer",
        .flag           = EBufferFlag::AllowDisplay,
        .width          = static_cast<uint32_t>(1080),
        .height         = static_cast<uint32_t>(720),
        .stride_in_byte = sizeof(float) * 4};
    buf_mngr->AllocBuffer(test_buffer_desc);

    BufferDesc test_buffer_desc2{
        .name           = "test buffer2",
        .flag           = EBufferFlag::AllowDisplay,
        .width          = static_cast<uint32_t>(5),
        .height         = static_cast<uint32_t>(5),
        .stride_in_byte = sizeof(float) * 4};
    buf_mngr->AllocBuffer(test_buffer_desc2);

    util::Singleton<Event::Center>::instance()
        ->Send(Gui::Event::CanvasDisplayTargetChange, {std::string{"default buffer"}});
}

void CudaPass::OnRun() noexcept {
    m_frame_cnt++;
    auto buf_mngr = util::Singleton<BufferManager>::instance();
    {
        auto frame_buffer = buf_mngr->GetBuffer("default buffer");
        m_output.SetData(frame_buffer->cuda_ptr, frame_buffer->desc.height * frame_buffer->desc.width);
        SetColor(uint2{frame_buffer->desc.width, frame_buffer->desc.height}, m_frame_cnt, m_output, m_stream.Get());
    }
    {
        auto frame_buffer = buf_mngr->GetBuffer("test buffer");
        m_output.SetData(frame_buffer->cuda_ptr, frame_buffer->desc.height * frame_buffer->desc.width);
        SetColor(uint2{frame_buffer->desc.width, frame_buffer->desc.height}, m_frame_cnt, m_output, m_stream.Get());
    }
    {
        auto frame_buffer = buf_mngr->GetBuffer("test buffer2");
        m_output.SetData(frame_buffer->cuda_ptr, frame_buffer->desc.height * frame_buffer->desc.width);
        SetColor(uint2{frame_buffer->desc.width, frame_buffer->desc.height}, m_frame_cnt, m_output, m_stream.Get());
    }
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
}