#pragma once

#include <optix.h>

#include <type_traits>
#include <vector>
#include <string>

#include "check.h"
#include "cuda/util.h"

namespace Pupil::optix {
template<typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template<typename T>
concept SBTTypes =
    requires {
        typename T::RayGenDataType;
        typename T::MissDataType;
        typename T::HitGroupDataType;
        typename T::CallablesDataType;
        typename T::ExceptionDataType;
    };

struct EmptyData {};

struct EmptySBT {
    using RayGenDataType = EmptyData;
    using MissDataType = EmptyData;
    using HitGroupDataType = EmptyData;
    using CallablesDataType = EmptyData;
    using ExceptionDataType = EmptyData;
};

template<typename T, typename U>
struct ProgDataPair {
    using DataType = U;
    T program;
    DataType data;
};

template<typename T>
struct ProgDataPair<T, void> {
    using DataType = EmptyData;
    T program;
    DataType data;
};

template<typename U>
using ProgDataDescPair = ProgDataPair<std::string, U>;
template<typename U>
using ProgDataBindingPair = ProgDataPair<OptixProgramGroup, U>;

template<SBTTypes T>
struct SBTDesc {
    ProgDataDescPair<typename T::RayGenDataType> ray_gen_data;
    std::vector<ProgDataDescPair<typename T::MissDataType>> miss_datas;
    std::vector<ProgDataDescPair<typename T::HitGroupDataType>> hit_datas;
    std::vector<ProgDataDescPair<typename T::CallablesDataType>> callables_datas;
    ProgDataDescPair<typename T::ExceptionDataType> exception_data;
};

template<typename U>
struct BindingInfo {
    using BindingPair = ProgDataBindingPair<U>;
    std::vector<BindingPair> datas;
};

template<SBTTypes T>
struct SBT {
private:
    CUdeviceptr m_ray_gen_record = 0;
    CUdeviceptr m_miss_record = 0;
    CUdeviceptr m_hitgroup_record = 0;
    CUdeviceptr m_callables_record = 0;
    CUdeviceptr m_exception_record = 0;

public:
    OptixShaderBindingTable sbt{};

    SBT(const SBTDesc<T> &desc, const Pipeline *pipeline)
    noexcept;

    using RayGenBindingType = BindingInfo<typename T::RayGenDataType>;
    using MissBindingType = BindingInfo<typename T::MissDataType>;
    using HitGroupBindingType = BindingInfo<typename T::HitGroupDataType>;
    using CallablesBindingType = BindingInfo<typename T::CallablesDataType>;
    using ExceptionBindingType = BindingInfo<typename T::ExceptionDataType>;

    using RayGenDataRecord = Record<typename RayGenBindingType::BindingPair::DataType>;
    using MissDataRecord = Record<typename MissBindingType::BindingPair::DataType>;
    using HitGroupDataRecord = Record<typename HitGroupBindingType::BindingPair::DataType>;
    using CallablesDataRecord = Record<typename CallablesBindingType::BindingPair::DataType>;
    using ExceptionDataRecord = Record<typename ExceptionBindingType::BindingPair::DataType>;

    void SetRayGenData(const RayGenBindingType &binding_info) noexcept;
    void SetMissData(const MissBindingType &binding_info) noexcept;
    void SetHitGroupData(const HitGroupBindingType &binding_info) noexcept;
    void SetCallablesData(const CallablesBindingType &binding_info) noexcept;
    void SetExceptionData(const ExceptionBindingType &binding_info) noexcept;

    void UpdateRayGenRecord(const RayGenDataRecord &record) noexcept;
    void UpdateMissRecord(const MissDataRecord &record, unsigned int offset) noexcept;
    void UpdateHitGroupRecord(const HitGroupDataRecord &record, unsigned int offset) noexcept;
    void UpdateCallablesRecord(const CallablesDataRecord &record, unsigned int offset) noexcept;
    void UpdateExceptionRecord(const ExceptionDataRecord &record) noexcept;

    void UpdateMissRecords(const MissDataRecord *records, unsigned int cnt, unsigned int offset) noexcept;
    void UpdateHitGroupRecords(const HitGroupDataRecord *records, unsigned int cnt, unsigned int offset) noexcept;
    void UpdateCallablesRecords(const CallablesDataRecord *records, unsigned int cnt, unsigned int offset) noexcept;

    ~SBT() noexcept;
};

}// namespace Pupil::optix

#include "sbt.inl"
