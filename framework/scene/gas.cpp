#include "gas.h"
#include "resource/shape.h"
#include "optix/check.h"
#include "optix/context.h"

#include "cuda/stream.h"
#include "cuda/check.h"
#include "cuda/vec_math.h"

#include <unordered_map>

#include <optix_stubs.h>

namespace Pupil {
    GAS::GAS(const util::CountableRef<resource::Shape>& shape) noexcept
        : m_shape(shape), m_handle(0), m_buffer(0), m_compacted_gas_size_host_p(nullptr) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&m_compacted_gas_size_host_p), sizeof(size_t), cudaHostAllocDefault));
        Update();
    }

    GAS::~GAS() noexcept {
        m_shape.Reset();

        auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::None);
        CUDA_FREE_ASYNC(m_buffer, *stream);
        CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(m_compacted_gas_size_host_p)));
    }

    void GAS::Update() noexcept {
        OptixBuildInput input = m_shape->GetOptixBuildInput();

        auto                   context = Pupil::util::Singleton<Pupil::optix::Context>::instance();
        OptixAccelBuildOptions accel_options{};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                                   OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                   OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        m_shape->WaitForDataUploading();
        util::Singleton<cuda::StreamManager>::instance()->Synchronize(cuda::EStreamTaskType::ShapeUploading);

        OptixAccelBufferSizes gas_buffer_sizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(*context, &accel_options, &input, 1u, &gas_buffer_sizes));

        auto        stream        = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::GASCreation);
        CUdeviceptr d_temp_buffer = 0;
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes, *stream));
        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size{};
        size_t      compacted_size_offset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compacted_size_offset + 8, *stream));
        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compacted_size_offset + 8));

        OptixAccelEmitDesc emit_property = {};
        emit_property.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_property.result             = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compacted_size_offset);

        OPTIX_CHECK(optixAccelBuild(
            *context,
            *stream,
            &accel_options,
            &input,
            1,
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size,
            gas_buffer_sizes.outputSizeInBytes,
            &m_handle,
            &emit_property,// emitted property list
            1              // num emitted properties
            ));

        CUDA_CHECK(cudaMemcpyAsync(m_compacted_gas_size_host_p, (void*)emit_property.result, sizeof(size_t), cudaMemcpyDeviceToHost, *stream));
        CUDA_FREE_ASYNC(m_buffer, *stream);
        stream->Synchronize();
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_buffer), *m_compacted_gas_size_host_p, *stream));
        OPTIX_CHECK(optixAccelCompact(*context, *stream, m_handle, m_buffer, *m_compacted_gas_size_host_p, &m_handle));
        CUDA_FREE_ASYNC(d_buffer_temp_output_gas_and_compacted_size, *stream);
        CUDA_FREE_ASYNC(d_temp_buffer, *stream);
    }

    struct GASManager::Impl {
        std::unordered_map<resource::Shape*, util::Data<GAS>> gas_pool;
    };

    GASManager::GASManager() noexcept {
        if (m_impl) return;
        m_impl = new Impl();
    }

    GASManager::~GASManager() noexcept {
    }

    util::CountableRef<GAS> GASManager::GetGAS(const util::CountableRef<resource::Shape>& shape) noexcept {
        if (auto it = m_impl->gas_pool.find(shape.Get());
            it != m_impl->gas_pool.end()) return it->second.GetRef();

        auto gas = util::Data<GAS>(std::make_unique<GAS>(shape));
        auto ref = gas.GetRef();
        m_impl->gas_pool.emplace(shape.Get(), std::move(gas));
        return ref;
    }

    void GASManager::Clear() noexcept {
        for (auto it = m_impl->gas_pool.begin(); it != m_impl->gas_pool.end();) {
            if (it->second.GetRefCount() == 0) {
                it = m_impl->gas_pool.erase(it);
            } else
                ++it;
        }
    }
}// namespace Pupil