#pragma once

#include <optix.h>

#include <type_traits>
#include <vector>
#include <string>

namespace Pupil::optix {
    template<typename T>
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    template<>
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record<void> {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    class Pipeline;

    class SBT {
    public:
        SBT(Pipeline* pipeline = nullptr)
        noexcept;
        ~SBT() noexcept;

        void SetPipeline(Pipeline* pipeline) noexcept;

        operator OptixShaderBindingTable() const noexcept;

        template<typename T>
        void SetRayGenRecord() noexcept { SetRayGenRecord(sizeof(Record<T>), sizeof(Record<T>::header)); }

        template<typename T>
        void SetHitgroupRecord(unsigned int num) noexcept { SetHitgroupRecord(num, sizeof(Record<T>), sizeof(Record<T>::header)); }

        template<typename T>
        void SetMissRecord(unsigned int num) noexcept { SetMissRecord(num, sizeof(Record<T>), sizeof(Record<T>::header)); }

        template<typename T>
        void SetCallablesRecord(unsigned int num) noexcept { SetCallablesRecord(num, sizeof(Record<T>), sizeof(Record<T>::header)); }

        template<typename T>
        void SetExceptionRecord() noexcept { SetExceptionRecord(sizeof(Record<T>), sizeof(Record<T>::header)); }

        void BindData(std::string_view program_name, void* data, unsigned int offset = 0, unsigned int num = 1) noexcept;
        void Finish() noexcept;

        void UpdateRayGenRecordData(void* data) noexcept;
        void UpdateHitgroupRecordData(void* data, unsigned int offset = 0, unsigned int num = 1) noexcept;
        void UpdateMissRecordData(void* data, unsigned int offset = 0, unsigned int num = 1) noexcept;
        void UpdateCallablesRecordData(void* data, unsigned int offset = 0, unsigned int num = 1) noexcept;
        void UpdateExceptionRecordData(void* data) noexcept;

    private:
        void SetRayGenRecord(size_t stride_in_byte, size_t record_header_size_in_byte) noexcept;
        void SetHitgroupRecord(unsigned int num, size_t stride_in_byte, size_t record_header_size_in_byte) noexcept;
        void SetMissRecord(unsigned int num, size_t stride_in_byte, size_t record_header_size_in_byte) noexcept;
        void SetCallablesRecord(unsigned int num, size_t stride_in_byte, size_t record_header_size_in_byte) noexcept;
        void SetExceptionRecord(size_t stride_in_byte, size_t record_header_size_in_byte) noexcept;

        struct Impl;
        Impl* m_impl = nullptr;
    };

}// namespace Pupil::optix
