#include "parser.h"
#include "tag.h"
#include "visitor.h"

#include <memory>
#include <array>
#include <filesystem>
#include <functional>
#include <assert.h>

namespace scene::xml {

std::unique_ptr<std::unordered_map<std::string, ETag>> s_tag_map = nullptr;
// std::unique_ptr<std::unordered_map<std::string, VisitorFunc>> s_visitor = nullptr;

void RegisterContext() {
    s_tag_map = std::make_unique<std::unordered_map<std::string, ETag>>();
    for (unsigned int i = 1u; i < (unsigned int)ETag::COUNT; i++) {
        s_tag_map->emplace(std::string{ S_TAGS_NAME[i - 1] }, static_cast<ETag>(i));
    }
}

// void DeregisterContext() {
//     s_tag_map.reset();
// }

void DfsParse(Parser *parser, pugi::xml_node node) {
    auto tag = s_tag_map->operator[](node.name());
    bool flag = Visit(tag, parser, node);
    if (!flag) return;
    for (pugi::xml_node &ch : node.children())
        DfsParse(parser, ch);
}

// void Parser::DeregisterContext() noexcept {
//     DeregisterContext();
// }

Parser::Parser() noexcept {
    if (s_tag_map == nullptr) {
        RegisterContext();
    }
}

Parser::~Parser() noexcept {
}

void Parser::LoadFromFile(std::string_view path) noexcept {
    std::filesystem::path file_path(path.data());

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(file_path.c_str());
    //assert(result == 0);
    DfsParse(this, doc.document_element());
}

void Parser::AddGlobalParam(std::string name, std::string value) noexcept {
    m_global_params[name] = value;
}

void Parser::ReplaceDefaultValue(pugi::xml_node *node) noexcept {
    for (auto attr : node->attributes()) {
        std::string a_value = attr.value();
        if (a_value.find('$') == std::string::npos)
            continue;
        for (auto &[p_name, p_value] : m_global_params) {
            size_t pos = 0;
            std::string temp_name = "$" + p_name;
            while ((pos = a_value.find(temp_name, pos)) != std::string::npos) {
                a_value.replace(pos, temp_name.length(), p_value);
                pos += p_value.length();
            }
        }
        attr.set_value(a_value.c_str());
    }
}
}// namespace scene::xml