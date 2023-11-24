#pragma once

#include "cuda/stream.h"
#include "cuda/util.h"

void SetColor(uint2, unsigned int, Pupil::cuda::RWArrayView<float4>&, Pupil::cuda::Stream*);