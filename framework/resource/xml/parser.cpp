#include "parser.h"
#include "tag.h"
#include "object.h"
#include "visitor.h"

#include <memory>
#include <array>
#include <filesystem>
#include <functional>

namespace Pupil::resource::xml {

std::unique_ptr<std::unordered_map<std::string, ETag>> s_tag_map = nullptr;

void RegisterContext() {
    s_tag_map = std::make_unique<std::unordered_map<std::string, ETag>>();
    for (unsigned int i = 1u; i < (unsigned int)ETag::_count; i++) {
        s_tag_map->emplace(std::string{ S_TAGS_NAME[i - 1] }, static_cast<ETag>(i));
    }
}

void DfsParse(Parser *parser, pugi::xml_node node) {
    ETag tag = ETag::_unknown;
    if (s_tag_map->find(node.name()) != s_tag_map->end())
        tag = s_tag_map->operator[](node.name());

    auto xml_g_mngr = parser->GetXMLGlobalManager();
    auto obj_parent = xml_g_mngr->current_obj;

    bool flag = Visit(tag, xml_g_mngr, node);
    if (!flag) return;

    for (pugi::xml_node &ch : node.children())
        DfsParse(parser, ch);

    xml_g_mngr->current_obj = obj_parent;
}

void Parser::DeregisterContext() noexcept {
    s_tag_map.reset();
}

Parser::Parser() noexcept {
    if (s_tag_map == nullptr) {
        RegisterContext();
    }
    m_global_manager = std::make_unique<GlobalManager>();
}

Parser::~Parser() noexcept {
}

Object *Parser::LoadFromFile(std::filesystem::path file_path) noexcept {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(file_path.c_str());
    DfsParse(this, doc.document_element());

    return m_global_manager->objects_pool[0].get();
}
}// namespace Pupil::resource::xml