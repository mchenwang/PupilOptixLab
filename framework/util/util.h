#pragma once

#include <memory>
#include <vector>
#include <string>

namespace Pupil::util {
    template<typename T>
    class Singleton {
    public:
        static T* instance() {
            static const std::unique_ptr<T> instance = std::make_unique<T>();
            return instance.get();
        }

        Singleton(const Singleton&)           = delete;
        Singleton& operator=(const Singleton) = delete;

    protected:
        Singleton() {}
    };

    // struct is from "https://www.cppstories.com/2021/heterogeneous-access-cpp20/"
    struct StringHash {
        using is_transparent = void;
        [[nodiscard]] size_t operator()(const char* txt) const {
            return std::hash<std::string_view>{}(txt);
        }
        [[nodiscard]] size_t operator()(std::string_view txt) const {
            return std::hash<std::string_view>{}(txt);
        }
        [[nodiscard]] size_t operator()(const std::string& txt) const {
            return std::hash<std::string>{}(txt);
        }
    };
}// namespace Pupil::util