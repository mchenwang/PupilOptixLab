#include "post_process.h"
#include "post_process.cuh"
#include "cuda/util.h"
#include "cuda/stream.h"
#include "cuda/data_view.h"

#include "system.h"
#include "imgui.h"
#include "gui.h"

#include "scene/scene.h"

#include "util/event.h"

#include <array>

namespace {
constexpr std::array m_tone_mapping_name = {
    "None",
    "ACES"
};
int m_tone_mapping_type = 0;
bool m_use_gamma = true;
float m_gamma = 2.2f;

// inline Pupil::cuda::EPostProcessType GetPostProcessType() noexcept {
//     if (m_tone_mapping_type == 1) {
//         if (m_use_gamma)
//             return Pupil::cuda::EPostProcessType::ACES_TONE_MAPPING_WITH_GAMMA;
//         else
//             return Pupil::cuda::EPostProcessType::ACES_TONE_MAPPING_WITHOUT_GAMMA;
//     } else {
//         if (m_use_gamma) return Pupil::cuda::EPostProcessType::GAMMA_ONLY;
//     }
//     return Pupil::cuda::EPostProcessType::NONE;
// }
inline uint32_t GetPostProcessType() noexcept {
    if (m_tone_mapping_type == 1) {
        if (m_use_gamma)
            return 1;
        else
            return 2;
    } else {
        if (m_use_gamma) return 3;
    }
    return 0;
}
}// namespace

namespace Pupil {
void PostProcessPass::Init() noexcept {
    m_stream = std::make_unique<cuda::Stream>();
    CUDA_CHECK(cudaEventCreateWithFlags(&finished_event, cudaEventDisableTiming));
    m_init_flag = true;
}
void PostProcessPass::Destroy() noexcept {
    if (!IsInitialized()) return;
    m_stream.reset();
    CUDA_CHECK(cudaEventDestroy(finished_event));
}
void PostProcessPass::BeforeRunning() noexcept {
    if (!IsInitialized()) return;
    auto buffer_mngr = util::Singleton<BufferManager>::instance();
    input = buffer_mngr->GetBuffer(BufferManager::CUSTOM_OUTPUT_BUFFER);
    if (util::Singleton<GuiPass>::instance()->IsInitialized()) {
        output = util::Singleton<GuiPass>::instance()->GetCurrentRenderOutputBuffer();
    } else
        output = input;
}
void PostProcessPass::Run() noexcept {
    if (!IsInitialized()) return;
    cuda::ConstArrayView<float4> input_view;
    input_view.SetData(input->cuda_res.ptr, m_image_size);
    cuda::RWArrayView<float4> output_view;
    output_view.SetData(output->cuda_res.ptr, m_image_size);
    PostProcessXX();
    // PostProcess(
    //     m_stream->GetStream(), finished_event,
    //     output_view.GetDataPtr(), input_view.GetDataPtr(),
    //     make_uint2(m_image_w, m_image_h),
    //     m_gamma,
    //     GetPostProcessType());
    m_stream->Synchronize();
    EventDispatcher<SystemEvent::PostProcessFinished>();
}

void PostProcessPass::SetScene(scene::Scene *scene) noexcept {
    if (!IsInitialized()) return;
    m_image_h = static_cast<uint32_t>(scene->sensor.film.h);
    m_image_w = static_cast<uint32_t>(scene->sensor.film.w);
    m_image_size = static_cast<size_t>(m_image_h) * m_image_w;
}

void PostProcessPass::SetToneMappingType(EToneMappingType type) noexcept {
    m_tone_mapping_type = static_cast<int>(type);
}
void PostProcessPass::SetGammaCorrection(bool use_gamma, float gamma) noexcept {
    m_use_gamma = use_gamma;
    if (m_use_gamma) m_gamma = gamma;
}

void PostProcessPass::Inspector() noexcept {
    if (!IsInitialized()) {
        ImGui::Text("Not Initialized.");
        return;
    }
    ImGui::Combo("Tone Mapping", &m_tone_mapping_type, m_tone_mapping_name.data(), (int)m_tone_mapping_name.size());
    ImGui::Checkbox("Gamma Correction", &m_use_gamma);
}
}// namespace Pupil