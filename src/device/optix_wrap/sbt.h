#pragma once

#include <optix.h>

#include <type_traits>
#include <vector>
#include <string>

#include "../error_handle.h"
#include "cuda_util/util.h"

namespace optix_wrap {
template<typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template<typename T>
struct RecordTraits {};

template<typename T>
struct RecordTraits<Record<T>> {
    using Type = T;
};
template<>
struct RecordTraits<void> {
    using Type = void;
};

template<typename T>
concept RecordType = requires { RecordTraits<T>::Type; };

template<typename T>
concept SBTTypes = requires {
                       typename T::RayGenDataType;
                       typename T::MissDataType;
                       typename T::HitGroupDataType;
                   };

template<SBTTypes T>
struct SBTDesc {
    template<typename U>
    struct Pair {
        std::string program_name;
        U data;
    };
    Pair<typename T::RayGenDataType> ray_gen_data;
    std::vector<Pair<typename T::MissDataType>> miss_datas;
    std::vector<Pair<typename T::HitGroupDataType>> hit_datas;
};

template<typename U>
struct BindingInfo {
    struct Pair {
        OptixProgramGroup program;
        U data;
    };
    std::vector<Pair> datas;
};

template<SBTTypes T>
struct SBT {
private:
    CUdeviceptr ray_gen_sbt = 0;
    CUdeviceptr miss_sbt = 0;
    CUdeviceptr hitgroup_sbt = 0;

public:
    OptixShaderBindingTable sbt{};

    SBT()
    noexcept;

    void SetRayGenData(BindingInfo<typename T::RayGenDataType> binding_info) noexcept;
    void SetMissData(BindingInfo<typename T::MissDataType> binding_info) noexcept;
    void SetHitGroupData(BindingInfo<typename T::HitGroupDataType> binding_info) noexcept;

    ~SBT() noexcept;
};

}// namespace optix_wrap

// implement
#include "optix_stubs.h"

namespace optix_wrap {
template<SBTTypes T>
SBT<T>::SBT() noexcept {
}

template<SBTTypes T>
void SBT<T>::SetRayGenData(BindingInfo<typename T::RayGenDataType> binding_info) noexcept {
    if constexpr (std::is_void_v<typename T::RayGenDataType>)
        return;
    else {
        using RayGenDataRecord = Record<typename T::RayGenDataType>;
        constexpr auto size = sizeof(RayGenDataRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ray_gen_sbt), size));
        RayGenDataRecord record{};

        OPTIX_CHECK(optixSbtRecordPackHeader(binding_info.datas[0].program, &record));
        record.data = binding_info.datas[0].data;
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(ray_gen_sbt),
            &record,
            size,
            cudaMemcpyHostToDevice));
        sbt.raygenRecord = ray_gen_sbt;
    }
}

template<SBTTypes T>
void SBT<T>::SetMissData(BindingInfo<typename T::MissDataType> binding_info) noexcept {
    if constexpr (std::is_void_v<typename T::MissDataType>)
        return;
    else {
        using MissDataRecord = Record<typename T::MissDataType>;
        const auto size = sizeof(MissDataRecord) * binding_info.datas.size();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_sbt), size));

        std::vector<MissDataRecord> ms_data;
        for (auto &[program, data] : binding_info.datas) {
            MissDataRecord record{};
            ms_data.push_back(record);
            OPTIX_CHECK(optixSbtRecordPackHeader(program, &ms_data.back()));
            ms_data.back().data = data;
        }

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(miss_sbt),
            ms_data.data(),
            size,
            cudaMemcpyHostToDevice));

        sbt.missRecordBase = miss_sbt;
        sbt.missRecordCount = static_cast<unsigned int>(binding_info.datas.size());
        sbt.missRecordStrideInBytes = sizeof(MissDataRecord);
    }
}

template<SBTTypes T>
void SBT<T>::SetHitGroupData(BindingInfo<typename T::HitGroupDataType> binding_info) noexcept {
    if constexpr (std::is_void_v<typename T::HitGroupDataType>)
        return;
    else {
        using HitGroupDataRecord = Record<typename T::HitGroupDataType>;
        const auto size = sizeof(HitGroupDataRecord) * binding_info.datas.size();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_sbt), size));

        std::vector<HitGroupDataRecord> hit_data;
        for (auto &[program, data] : binding_info.datas) {
            HitGroupDataRecord record{};
            hit_data.push_back(record);
            OPTIX_CHECK(optixSbtRecordPackHeader(program, &hit_data.back()));
            hit_data.back().data = data;
        }

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(hitgroup_sbt),
            hit_data.data(),
            size,
            cudaMemcpyHostToDevice));

        sbt.hitgroupRecordBase = hitgroup_sbt;
        sbt.hitgroupRecordCount = static_cast<unsigned int>(binding_info.datas.size());
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupDataRecord);
    }
}

template<SBTTypes T>
SBT<T>::~SBT() noexcept {
    CUDA_FREE(sbt.raygenRecord);
    CUDA_FREE(sbt.missRecordBase);
    CUDA_FREE(sbt.hitgroupRecordBase);
}
}// namespace optix_wrap