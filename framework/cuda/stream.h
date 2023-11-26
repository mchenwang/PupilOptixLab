#pragma once

#include "util/util.h"
#include "util/data.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace Pupil::cuda {
    class Stream {
    public:
        Stream(std::string_view name = "", bool non_blocking = true) noexcept;
        ~Stream() noexcept;

        operator cudaStream_t() const noexcept { return m_stream; }

        cudaStream_t GetStream() const noexcept { return m_stream; }

        void Synchronize() noexcept;

        void        SetName(std::string_view name) noexcept { m_name = name; }
        std::string GetName() const noexcept { return m_name; }

    private:
        std::string  m_name;
        cudaStream_t m_stream;
    };

    class Event {
    public:
        Event(bool use_timer = false) noexcept;
        ~Event() noexcept;

        operator cudaEvent_t() const noexcept { return m_event; }

        void Record(Stream* stream = nullptr) noexcept;

        void Synchronize() noexcept;
        bool IsCompleted() noexcept;
        bool IsRecorded() noexcept { return m_record_flag; }

    private:
        cudaEvent_t m_event;
        bool        m_record_flag;
    };

    /**
     * None ---------------------------------------------------------------->
     * 
     * Shape --> GAS --> IAS ----------------|
     *       \                               |                   Downloading
     *        --------------->               |                  /
     *                         \           Global------> Render ---->
     * Texture --> Material --> SBT ---------|------|
     *         \                             |
     *          ------> Emitter -------------|
    */
    enum class EStreamTaskType : unsigned int {
        Render            = 1 << 0,
        ShapeUploading    = 1 << 1,
        TextureUploading  = 1 << 2,
        MaterialUploading = 1 << 3,
        EmitterUploading  = 1 << 4,
        SBTUploading      = 1 << 5,
        GlobalUploading   = 1 << 6,
        Downloading       = 1 << 7,

        GASCreation    = 1 << 8,
        IASCreation    = 1 << 9,
        BufferCreation = 1 << 10,
        GUIInterop     = 1 << 11,

        None      = 1 << 12,
        Custom_1  = 1 << 13,
        Custom_2  = 1 << 14,
        Custom_3  = 1 << 15,
        Custom_4  = 1 << 16,
        Custom_5  = 1 << 17,
        Custom_6  = 1 << 18,
        Custom_7  = 1 << 19,
        Custom_8  = 1 << 20,
        Custom_9  = 1 << 21,
        Custom_10 = 1 << 22,

        SceneUploading = ShapeUploading | TextureUploading | MaterialUploading | EmitterUploading,
        SystemTask     = ((1ll << 12) - 1),
        ALL            = ((1ll << 32) - 1)
    };

    inline bool operator&(EStreamTaskType lhs, EStreamTaskType rhs) {
        return static_cast<unsigned int>(lhs) & static_cast<unsigned int>(rhs);
    }

    inline EStreamTaskType operator|(EStreamTaskType lhs, EStreamTaskType rhs) {
        return static_cast<EStreamTaskType>(static_cast<unsigned int>(lhs) | static_cast<unsigned int>(rhs));
    }

    class StreamManager : public util::Singleton<StreamManager> {
    public:
        StreamManager() noexcept;
        ~StreamManager() noexcept;

        /**
         * @param task_type specific type of task
         * @note task only has one type
        */
        util::CountableRef<Stream> Alloc(EStreamTaskType task_type, std::string_view name = "", bool non_blocking = true) noexcept;
        /**
         * @param task_type specific types of tasks
         * @note multi-types can be synchronized at the same time
        */
        void Synchronize(EStreamTaskType task_type) noexcept;

        void Destroy() noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil::cuda