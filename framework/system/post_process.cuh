#pragma once
#include "cuda/preprocessor.h"
#include "cuda/vec_math.h"
#include "cuda/data_view.h"

namespace Pupil::cuda {
enum class EPostProcessType {
    NONE,
    ACES_TONE_MAPPING_WITH_GAMMA,
    ACES_TONE_MAPPING_WITHOUT_GAMMA,
    GAMMA_ONLY
};
void PostProcess(
    cudaStream_t stream, cudaEvent_t finished_event,
    cuda::RWArrayView<float4> &output_image,
    cuda::ConstArrayView<float4> &input_image,
    uint2 size, float gamma = 2.2f,
    EPostProcessType post_process_type = EPostProcessType::NONE);
}// namespace Pupil::cuda