#pragma once

#include <string>
#include <unordered_map>
#include <memory>

namespace pugi {
class xml_node;
}

namespace scene {
struct Scene;

namespace xml {
class Parser {
private:
    std::unordered_map<std::string, std::string> m_global_params;

public:
    std::unique_ptr<scene::Scene> scene = nullptr;

    Parser() noexcept;
    ~Parser() noexcept;

    // static void DeregisterContext() noexcept;

    void LoadFromFile(std::string_view path) noexcept;
    void AddGlobalParam(std::string, std::string) noexcept;

    void ReplaceDefaultValue(pugi::xml_node *) noexcept;
};
}// namespace xml
}// namespace scene