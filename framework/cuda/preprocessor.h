#pragma once

// clang-format off
#if defined(__CUDACC__) || defined(__CUDABE__)
#   if !defined(PUPIL_OPTIX) && !defined(PUPIL_CUDA)
#		define PUPIL_CUDA
#	endif
#   if defined(PUPIL_CPP)
#		undef PUPIL_CPP
#	endif
#   define CUDA_HOST __host__
#   define CUDA_DEVICE __device__
#   define CUDA_HOSTDEVICE __host__ __device__
#   define CUDA_GLOBAL __global__
#   define CUDA_INLINE __forceinline__
#   define CONST_STATIC_INIT(...)
#else
#   if defined(PUPIL_OPTIX)
#		undef PUPIL_OPTIX
#	endif
#   if defined(PUPIL_CUDA)
#		undef PUPIL_CUDA
#	endif
#   define CUDA_HOST
#   define CUDA_DEVICE
#   define CUDA_HOSTDEVICE
#   define CUDA_GLOBAL
#   define CUDA_INLINE inline
#   define CONST_STATIC_INIT(...) = __VA_ARGS__
#endif
// clang-format on
