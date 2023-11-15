#include "sbt.h"
#include "pipeline.h"
#include "check.h"

#include "cuda/check.h"
#include "cuda/stream.h"

#include <optix_stubs.h>

namespace Pupil::optix {
    struct SBT::Impl {
        OptixShaderBindingTable sbt{};

        Pipeline* pipeline;

        struct {
            char*        data        = nullptr;
            size_t       record_size = 0;
            size_t       header_size = 0;
            unsigned int record_num  = 0;
        } ray_gen, hitgroup, miss, callables, exception;

        void* GetRecordPtr(decltype(ray_gen) record, unsigned int offset = 0) {
            return reinterpret_cast<void*>(
                reinterpret_cast<size_t>(record.data) + offset * record.record_size);
        }

        void* GetRecordDataPtr(decltype(ray_gen) record, unsigned int offset = 0) {
            return reinterpret_cast<void*>(
                reinterpret_cast<size_t>(record.data) + offset * record.record_size + record.header_size);
        }
    };

    SBT::SBT(Pipeline* pipeline) noexcept {
        m_impl           = new Impl();
        m_impl->pipeline = pipeline;

        SetRayGenRecord<void>();
    }

    void SBT::SetPipeline(Pipeline* pipeline) noexcept {
        m_impl->pipeline = pipeline;
    }

    SBT::~SBT() noexcept {
        auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::None);

        CUDA_FREE_ASYNC(m_impl->sbt.raygenRecord, *stream);
        CUDA_FREE_ASYNC(m_impl->sbt.missRecordBase, *stream);
        CUDA_FREE_ASYNC(m_impl->sbt.hitgroupRecordBase, *stream);
        CUDA_FREE_ASYNC(m_impl->sbt.callablesRecordBase, *stream);
        CUDA_FREE_ASYNC(m_impl->sbt.exceptionRecord, *stream);

        if (m_impl->ray_gen.data) free(m_impl->ray_gen.data);
        if (m_impl->hitgroup.data) free(m_impl->hitgroup.data);
        if (m_impl->miss.data) free(m_impl->miss.data);
        if (m_impl->callables.data) free(m_impl->callables.data);
        if (m_impl->exception.data) free(m_impl->exception.data);

        delete m_impl;
    }

    SBT::operator OptixShaderBindingTable() const noexcept {
        return m_impl->sbt;
    }

    void SBT::Finish() noexcept {
        auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::SBTUploading);

        // ray gen
        {
            if (m_impl->sbt.raygenRecord) CUDA_FREE_ASYNC(m_impl->sbt.raygenRecord, *stream);
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->sbt.raygenRecord), m_impl->ray_gen.record_size, *stream));
            CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void*>(m_impl->sbt.raygenRecord),
                reinterpret_cast<void*>(m_impl->ray_gen.data),
                m_impl->ray_gen.record_size,
                cudaMemcpyHostToDevice,
                *stream));
        }

        // hitgroup
        if (m_impl->hitgroup.data) {
            if (m_impl->sbt.hitgroupRecordBase) CUDA_FREE_ASYNC(m_impl->sbt.hitgroupRecordBase, *stream);
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->sbt.hitgroupRecordBase), m_impl->hitgroup.record_size * m_impl->hitgroup.record_num, *stream));
            CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void*>(m_impl->sbt.hitgroupRecordBase),
                reinterpret_cast<void*>(m_impl->hitgroup.data),
                m_impl->hitgroup.record_size * m_impl->hitgroup.record_num,
                cudaMemcpyHostToDevice,
                *stream));

            m_impl->sbt.hitgroupRecordCount         = m_impl->hitgroup.record_num;
            m_impl->sbt.hitgroupRecordStrideInBytes = m_impl->hitgroup.record_size;
        }

        // miss
        if (m_impl->miss.data) {
            if (m_impl->sbt.missRecordBase) CUDA_FREE_ASYNC(m_impl->sbt.missRecordBase, *stream);
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->sbt.missRecordBase), m_impl->miss.record_size * m_impl->miss.record_num, *stream));
            CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void*>(m_impl->sbt.missRecordBase),
                reinterpret_cast<void*>(m_impl->miss.data),
                m_impl->miss.record_size * m_impl->miss.record_num,
                cudaMemcpyHostToDevice,
                *stream));

            m_impl->sbt.missRecordCount         = m_impl->miss.record_num;
            m_impl->sbt.missRecordStrideInBytes = m_impl->miss.record_size;
        }

        // callables
        if (m_impl->callables.data) {
            if (m_impl->sbt.callablesRecordBase) CUDA_FREE_ASYNC(m_impl->sbt.callablesRecordBase, *stream);
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->sbt.callablesRecordBase), m_impl->callables.record_size * m_impl->callables.record_num, *stream));
            CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void*>(m_impl->sbt.callablesRecordBase),
                reinterpret_cast<void*>(m_impl->callables.data),
                m_impl->callables.record_size * m_impl->callables.record_num,
                cudaMemcpyHostToDevice,
                *stream));

            m_impl->sbt.callablesRecordCount         = m_impl->callables.record_num;
            m_impl->sbt.callablesRecordStrideInBytes = m_impl->callables.record_size;
        }

        // exception
        if (m_impl->exception.data) {
            if (m_impl->sbt.exceptionRecord) CUDA_FREE_ASYNC(m_impl->sbt.exceptionRecord, *stream);
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->sbt.exceptionRecord), m_impl->exception.record_size, *stream));
            CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void*>(m_impl->sbt.exceptionRecord),
                reinterpret_cast<void*>(m_impl->exception.data),
                m_impl->exception.record_size,
                cudaMemcpyHostToDevice,
                *stream));
        }
    }

    void SBT::BindData(std::string_view program_name, void* data, unsigned int offset, unsigned int num) noexcept {
        auto program = m_impl->pipeline->FindProgram(program_name);
        switch (program->GetKind()) {
            case OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN: {
                auto record = m_impl->GetRecordPtr(m_impl->ray_gen);
                OPTIX_CHECK(optixSbtRecordPackHeader(*program, record));
                auto record_data = m_impl->GetRecordDataPtr(m_impl->ray_gen);
                if (data) memcpy(record_data, data, m_impl->ray_gen.record_size - m_impl->ray_gen.header_size);
            } break;
            case OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP: {
                for (auto i = 0u; i < num; i++) {
                    auto record = m_impl->GetRecordPtr(m_impl->hitgroup, offset + i);
                    OPTIX_CHECK(optixSbtRecordPackHeader(*program, record));
                    auto record_data = m_impl->GetRecordDataPtr(m_impl->hitgroup, offset + i);
                    if (data) memcpy(record_data, data, m_impl->hitgroup.record_size - m_impl->hitgroup.header_size);
                }
            } break;
            case OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS: {
                for (auto i = 0u; i < num; i++) {
                    auto record = m_impl->GetRecordPtr(m_impl->miss, offset + i);
                    OPTIX_CHECK(optixSbtRecordPackHeader(*program, record));
                    auto record_data = m_impl->GetRecordDataPtr(m_impl->miss, offset + i);
                    if (data) memcpy(record_data, data, m_impl->miss.record_size - m_impl->miss.header_size);
                }
            } break;
            case OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_CALLABLES: {
                for (auto i = 0u; i < num; i++) {
                    auto record = m_impl->GetRecordPtr(m_impl->callables, offset + i);
                    OPTIX_CHECK(optixSbtRecordPackHeader(*program, record));
                    auto record_data = m_impl->GetRecordDataPtr(m_impl->callables, offset + i);
                    if (data) memcpy(record_data, data, m_impl->callables.record_size - m_impl->callables.header_size);
                }
            } break;
            case OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_EXCEPTION: {
                auto record = m_impl->GetRecordPtr(m_impl->exception);
                OPTIX_CHECK(optixSbtRecordPackHeader(*program, record));
                auto record_data = m_impl->GetRecordDataPtr(m_impl->exception);
                if (data) memcpy(record_data, data, m_impl->exception.record_size - m_impl->exception.header_size);
            } break;
        }
    }

    void SBT::UpdateRayGenRecordData(void* data) noexcept {
        auto record_data = m_impl->GetRecordDataPtr(m_impl->ray_gen);
        memcpy(record_data, data, m_impl->ray_gen.record_size - m_impl->ray_gen.header_size);
    }

    void SBT::UpdateHitgroupRecordData(void* data, unsigned int offset, unsigned int num) noexcept {
        for (auto i = 0u; i < num; i++) {
            auto record_data = m_impl->GetRecordDataPtr(m_impl->hitgroup, offset + i);
            memcpy(record_data, data, m_impl->hitgroup.record_size - m_impl->hitgroup.header_size);
        }
    }

    void SBT::UpdateMissRecordData(void* data, unsigned int offset, unsigned int num) noexcept {
        for (auto i = 0u; i < num; i++) {
            auto record_data = m_impl->GetRecordDataPtr(m_impl->miss, offset + i);
            memcpy(record_data, data, m_impl->miss.record_size - m_impl->miss.header_size);
        }
    }

    void SBT::UpdateCallablesRecordData(void* data, unsigned int offset, unsigned int num) noexcept {
        for (auto i = 0u; i < num; i++) {
            auto record_data = m_impl->GetRecordDataPtr(m_impl->callables, offset + i);
            memcpy(record_data, data, m_impl->callables.record_size - m_impl->callables.header_size);
        }
    }

    void SBT::UpdateExceptionRecordData(void* data) noexcept {
        auto record_data = m_impl->GetRecordDataPtr(m_impl->exception);
        memcpy(record_data, data, m_impl->exception.record_size - m_impl->exception.header_size);
    }

    void SBT::SetRayGenRecord(size_t stride_in_byte, size_t record_header_size_in_byte) noexcept {
        m_impl->ray_gen.record_size = stride_in_byte;
        m_impl->ray_gen.header_size = record_header_size_in_byte;
        m_impl->ray_gen.record_num  = 1;
        if (m_impl->ray_gen.data) free(m_impl->ray_gen.data);
        m_impl->ray_gen.data = reinterpret_cast<char*>(malloc(stride_in_byte));
        memset(reinterpret_cast<void*>(m_impl->ray_gen.data), 0, stride_in_byte);
    }

    void SBT::SetHitgroupRecord(unsigned int num, size_t stride_in_byte, size_t record_header_size_in_byte) noexcept {
        m_impl->hitgroup.record_size = stride_in_byte;
        m_impl->hitgroup.header_size = record_header_size_in_byte;
        m_impl->hitgroup.record_num  = num;
        if (m_impl->hitgroup.data) free(m_impl->hitgroup.data);
        m_impl->hitgroup.data = reinterpret_cast<char*>(malloc(stride_in_byte * num));
        memset(reinterpret_cast<void*>(m_impl->hitgroup.data), 0, stride_in_byte * num);
    }

    void SBT::SetMissRecord(unsigned int num, size_t stride_in_byte, size_t record_header_size_in_byte) noexcept {
        m_impl->miss.record_size = stride_in_byte;
        m_impl->miss.header_size = record_header_size_in_byte;
        m_impl->miss.record_num  = num;
        if (m_impl->miss.data) free(m_impl->miss.data);
        m_impl->miss.data = reinterpret_cast<char*>(malloc(stride_in_byte * num));
        memset(reinterpret_cast<void*>(m_impl->miss.data), 0, stride_in_byte * num);
    }

    void SBT::SetCallablesRecord(unsigned int num, size_t stride_in_byte, size_t record_header_size_in_byte) noexcept {
        m_impl->callables.record_size = stride_in_byte;
        m_impl->callables.header_size = record_header_size_in_byte;
        m_impl->callables.record_num  = num;
        if (m_impl->callables.data) free(m_impl->callables.data);
        m_impl->callables.data = reinterpret_cast<char*>(malloc(stride_in_byte * num));
        memset(reinterpret_cast<void*>(m_impl->callables.data), 0, stride_in_byte * num);
    }

    void SBT::SetExceptionRecord(size_t stride_in_byte, size_t record_header_size_in_byte) noexcept {
        m_impl->exception.record_size = stride_in_byte;
        m_impl->exception.header_size = record_header_size_in_byte;
        m_impl->exception.record_num  = 1;
        if (m_impl->exception.data) free(m_impl->exception.data);
        m_impl->exception.data = reinterpret_cast<char*>(malloc(stride_in_byte));
        memset(reinterpret_cast<void*>(m_impl->exception.data), 0, stride_in_byte);
    }

}// namespace Pupil::optix