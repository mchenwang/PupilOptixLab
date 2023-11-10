#pragma once

#include "util/util.h"
#include "util/data.h"

#include "resource/shape.h"

#include <optix.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

// namespace Pupil::resource {
//     struct Shape;
// }

namespace Pupil {
    class GAS {
    public:
        GAS(const util::CountableRef<resource::Shape>& shape)
        noexcept;
        ~GAS() noexcept;

        operator OptixTraversableHandle() const noexcept {
            return m_handle;
        }

        void Update() noexcept;

    private:
        util::CountableRef<resource::Shape> m_shape;
        OptixTraversableHandle              m_handle;
        CUdeviceptr                         m_buffer;
        size_t*                             m_compacted_gas_size_host_p;
    };

    class GASManager : util::Singleton<GASManager> {
    public:
        GASManager() noexcept;
        ~GASManager() noexcept;
        util::CountableRef<GAS> GetGAS(const util::CountableRef<resource::Shape>& shape) noexcept;

        void Clear() noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil