#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda/preprocessor.h"
#include "cuda/vec_math.h"
//#include "cuda/data_view.h"

// namespace Pupil {
// namespace cuda {
// enum class EPostProcessType {
//     NONE,
//     ACES_TONE_MAPPING_WITH_GAMMA,
//     ACES_TONE_MAPPING_WITHOUT_GAMMA,
//     GAMMA_ONLY
// };

//void PostProcess(
//    cudaStream_t stream, cudaEvent_t finished_event,
//    float4 *output_image, const float4 *input_image,
//    uint2 size, float gamma, unsigned int post_process_type);
// }
// }// namespace Pupil::cuda

void PostProcessXX();