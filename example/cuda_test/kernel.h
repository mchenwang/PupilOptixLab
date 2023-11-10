#pragma once

#include "cuda/preprocessor.h"
#include "cuda/stream.h"

void SetColor(uint2, unsigned int, Pupil::cuda::RWArrayView<float4>&, Pupil::cuda::Stream*);