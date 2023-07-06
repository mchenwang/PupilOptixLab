#pragma once

#include "cuda/preprocessor.h"
#include "cuda/stream.h"
#include "cuda/data_view.h"

void SetColor(unsigned int, Pupil::cuda::RWArrayView<float4> &, Pupil::cuda::Stream *);