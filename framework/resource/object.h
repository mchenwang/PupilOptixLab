#pragma once

#include <string>

namespace Pupil::resource {
    class Object {
    public:
        Object(std::string_view name) noexcept : m_name(name){};

        auto GetName() const noexcept { return m_name; }

        virtual void*            Clone() const noexcept               = 0;
        virtual std::string_view GetResourceType() const noexcept     = 0;
        virtual uint64_t         GetMemorySizeInByte() const noexcept = 0;

    protected:
        std::string m_name;
    };
}// namespace Pupil::resource