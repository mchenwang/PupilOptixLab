#pragma once

#include <memory>

namespace util {
template<typename T>
class Singleton {
public:
    static T *instance() {
        static const std::unique_ptr<T> instance = std::make_unique<T>();
        return instance.get();
    }

    Singleton(const Singleton &) = delete;
    Singleton &operator=(const Singleton) = delete;

protected:
    Singleton() {}
};
}// namespace util