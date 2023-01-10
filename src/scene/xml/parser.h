#pragma once

#include <string>
#include <memory>

namespace scene {
class Scene;

namespace xml {
struct Object;
struct GlobalManager;

class Parser {
private:
    std::unique_ptr<GlobalManager> m_global_manager;

public:
    Parser() noexcept;
    ~Parser() noexcept;

    static void DeregisterContext() noexcept;

    [[nodiscard]] Object *LoadFromFile(std::string_view path) noexcept;
    [[nodiscard]] GlobalManager *GetXMLGlobalManager() const noexcept { return m_global_manager.get(); }
};
}// namespace xml
}// namespace scene