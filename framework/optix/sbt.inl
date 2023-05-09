#include "optix_stubs.h"

namespace Pupil::optix {
template<SBTTypes T>
SBT<T>::SBT(const SBTDesc<T> &desc, const Pipeline *pipeline) noexcept {
    {
        typename SBT<T>::RayGenBindingType rg_data;
        typename decltype(rg_data)::BindingPair data{
            .program = pipeline->FindProgram(desc.ray_gen_data.program),
            .data = desc.ray_gen_data.data
        };
        rg_data.datas.push_back(data);
        SetRayGenData(rg_data);
    }
    {
        typename SBT<T>::HitGroupBindingType hit_datas{};
        for (auto &hit_data : desc.hit_datas) {
            typename decltype(hit_datas)::BindingPair data{
                .program = pipeline->FindProgram(hit_data.program),
                .data = hit_data.data
            };
            hit_datas.datas.push_back(data);
        }
        SetHitGroupData(hit_datas);
    }
    {
        typename SBT<T>::MissBindingType miss_datas{};
        for (auto &miss_data : desc.miss_datas) {
            typename decltype(miss_datas)::BindingPair data{
                .program = pipeline->FindProgram(miss_data.program),
                .data = miss_data.data
            };
            miss_datas.datas.push_back(data);
        }
        SetMissData(miss_datas);
    }
    {
        typename SBT<T>::CallablesBindingType call_datas{};
        for (auto &call_data : desc.callables_datas) {
            typename decltype(call_datas)::BindingPair data{
                .program = pipeline->FindProgram(call_data.program),
                .data = call_data.data
            };
            call_datas.datas.push_back(data);
        }
        SetCallablesData(call_datas);
    }
    {
        typename SBT<T>::ExceptionBindingType exception_data{};
        typename decltype(exception_data)::BindingPair data{
            .program = pipeline->FindProgram(desc.exception_data.program),
            .data = desc.exception_data.data
        };
        exception_data.datas.push_back(data);
        SetExceptionData(exception_data);
    }
}

template<SBTTypes T>
void SBT<T>::SetRayGenData(const RayGenBindingType &binding_info) noexcept {
    constexpr auto size = sizeof(RayGenDataRecord);

    if (m_ray_gen_record) CUDA_FREE(m_ray_gen_record);
    if (binding_info.datas[0].program) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_ray_gen_record), size));
        RayGenDataRecord record{};

        OPTIX_CHECK(optixSbtRecordPackHeader(binding_info.datas[0].program, &record));
        record.data = binding_info.datas[0].data;
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_ray_gen_record),
            &record,
            size,
            cudaMemcpyHostToDevice));
    }
    sbt.raygenRecord = m_ray_gen_record;
}

template<SBTTypes T>
void SBT<T>::SetMissData(const MissBindingType &binding_info) noexcept {
    const auto size = sizeof(MissDataRecord) * binding_info.datas.size();
    unsigned int record_cnt = 0u;

    if (m_miss_record) CUDA_FREE(m_miss_record);
    if (size != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_miss_record), size));

        std::vector<MissDataRecord> ms_data;
        for (auto &[program, data] : binding_info.datas) {
            if (program == nullptr) continue;
            MissDataRecord record{};
            ms_data.push_back(record);
            OPTIX_CHECK(optixSbtRecordPackHeader(program, &ms_data.back()));
            ms_data.back().data = data;
        }

        if (ms_data.size() > 0)
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(m_miss_record),
                ms_data.data(),
                size,
                cudaMemcpyHostToDevice));
        record_cnt = static_cast<unsigned int>(ms_data.size());
    } else {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_miss_record), sizeof(MissDataRecord)));
        record_cnt = 1;
    }
    sbt.missRecordBase = m_miss_record;
    sbt.missRecordCount = record_cnt;
    sbt.missRecordStrideInBytes = sizeof(MissDataRecord);
}

template<SBTTypes T>
void SBT<T>::SetHitGroupData(const HitGroupBindingType &binding_info) noexcept {
    const auto size = sizeof(HitGroupDataRecord) * binding_info.datas.size();
    unsigned int record_cnt = 0u;

    if (m_hitgroup_record) CUDA_FREE(m_hitgroup_record);
    if (size != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_hitgroup_record), size));

        std::vector<HitGroupDataRecord> hit_data;
        for (auto &[program, data] : binding_info.datas) {
            if (program == nullptr) continue;
            HitGroupDataRecord record{};
            hit_data.push_back(record);
            OPTIX_CHECK(optixSbtRecordPackHeader(program, &hit_data.back()));
            hit_data.back().data = data;
        }

        if (hit_data.size() > 0)
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(m_hitgroup_record),
                hit_data.data(),
                size,
                cudaMemcpyHostToDevice));
        record_cnt = static_cast<unsigned int>(hit_data.size());
    }
    sbt.hitgroupRecordBase = m_hitgroup_record;
    sbt.hitgroupRecordCount = record_cnt;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupDataRecord);
}

template<SBTTypes T>
void SBT<T>::SetCallablesData(const CallablesBindingType &binding_info) noexcept {
    const auto size = sizeof(CallablesDataRecord) * binding_info.datas.size();
    unsigned int record_cnt = 0u;

    if (m_callables_record) CUDA_FREE(m_callables_record);
    if (size != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_callables_record), size));

        std::vector<CallablesDataRecord> call_data;
        for (auto &[program, data] : binding_info.datas) {
            if (program == nullptr) continue;
            CallablesDataRecord record{};
            call_data.push_back(record);
            OPTIX_CHECK(optixSbtRecordPackHeader(program, &call_data.back()));
            call_data.back().data = data;
        }

        if (call_data.size() > 0)
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(m_callables_record),
                call_data.data(),
                size,
                cudaMemcpyHostToDevice));
        record_cnt = static_cast<unsigned int>(call_data.size());
    }
    sbt.callablesRecordBase = m_callables_record;
    sbt.callablesRecordCount = record_cnt;
    sbt.callablesRecordStrideInBytes = sizeof(CallablesDataRecord);
}

template<SBTTypes T>
void SBT<T>::SetExceptionData(const ExceptionBindingType &binding_info) noexcept {
    constexpr auto size = sizeof(ExceptionDataRecord);

    if (m_exception_record) CUDA_FREE(m_exception_record);
    if (binding_info.datas[0].program) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_exception_record), size));
        ExceptionDataRecord record{};

        OPTIX_CHECK(optixSbtRecordPackHeader(binding_info.datas[0].program, &record));
        record.data = binding_info.datas[0].data;
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_exception_record),
            &record,
            size,
            cudaMemcpyHostToDevice));
    }
    sbt.exceptionRecord = m_exception_record;
}

template<SBTTypes T>
void SBT<T>::UpdateRayGenRecord(const RayGenDataRecord &record) noexcept {
    constexpr auto size = sizeof(RayGenDataRecord);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(m_ray_gen_record),
        &record,
        size,
        cudaMemcpyHostToDevice));
}

template<SBTTypes T>
void SBT<T>::UpdateMissRecord(const MissDataRecord &record, unsigned int offset) noexcept {
    constexpr auto size = sizeof(MissDataRecord);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(m_miss_record + size * offset),
        &record,
        size,
        cudaMemcpyHostToDevice));
}

template<SBTTypes T>
void SBT<T>::UpdateHitGroupRecord(const HitGroupDataRecord &record, unsigned int offset) noexcept {
    constexpr auto size = sizeof(HitGroupDataRecord);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(m_hitgroup_record + size * offset),
        &record,
        size,
        cudaMemcpyHostToDevice));
}

template<SBTTypes T>
void SBT<T>::UpdateCallablesRecord(const CallablesDataRecord &record, unsigned int offset) noexcept {
    constexpr auto size = sizeof(CallablesDataRecord);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(m_callables_record + size * offset),
        &record,
        size,
        cudaMemcpyHostToDevice));
}

template<SBTTypes T>
void SBT<T>::UpdateExceptionRecord(const ExceptionDataRecord &record) noexcept {
    constexpr auto size = sizeof(ExceptionDataRecord);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(m_exception_record),
        &record,
        size,
        cudaMemcpyHostToDevice));
}

template<SBTTypes T>
void SBT<T>::UpdateMissRecords(const MissDataRecord *records, unsigned int cnt, unsigned int offset) noexcept {
    if (cnt < 1) return;
    constexpr auto size = sizeof(MissDataRecord);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(m_miss_record + size * offset),
        &records,
        size * cnt,
        cudaMemcpyHostToDevice));
}

template<SBTTypes T>
void SBT<T>::UpdateHitGroupRecords(const HitGroupDataRecord *records, unsigned int cnt, unsigned int offset) noexcept {
    if (cnt < 1) return;
    constexpr auto size = sizeof(HitGroupDataRecord);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(m_hitgroup_record + size * offset),
        &records,
        size * cnt,
        cudaMemcpyHostToDevice));
}

template<SBTTypes T>
void SBT<T>::UpdateCallablesRecords(const CallablesDataRecord *records, unsigned int cnt, unsigned int offset) noexcept {
    if (cnt < 1) return;
    constexpr auto size = sizeof(CallablesDataRecord);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(m_callables_record + size * offset),
        &records,
        size * cnt,
        cudaMemcpyHostToDevice));
}

template<SBTTypes T>
SBT<T>::~SBT() noexcept {
    CUDA_FREE(sbt.raygenRecord);
    CUDA_FREE(sbt.missRecordBase);
    CUDA_FREE(sbt.hitgroupRecordBase);
    CUDA_FREE(sbt.callablesRecordBase);
    CUDA_FREE(sbt.exceptionRecord);
}
}// namespace Pupil::optix