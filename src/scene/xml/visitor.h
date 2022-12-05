#pragma once

#include "tag.h"
#include "parser.h"
#include "scene/scene.h"

#include "pugixml.hpp"

#include <iostream>
#include <array>
#include <functional>

namespace scene::xml {
template<ETag T>
struct Visitor {
    bool operator()(Parser *parser, pugi::xml_node &node) {
        std::cout << node.name() << " skip\n";
        return false;
    }
};

#define IMPL_VISITOR(Tag, code)                                 \
    template<>                                                  \
    struct Visitor<Tag> {                                       \
        bool operator()(Parser *parser, pugi::xml_node &node) { \
            code                                                \
        }                                                       \
    };

// clang-format off
IMPL_VISITOR(ETag::_scene,
    std::cout << "read scene\n";
    return true;
)

IMPL_VISITOR(ETag::_default,
    parser->ReplaceDefaultValue(&node);
    std::string name = node.attribute("name").value();
    std::string value = node.attribute("value").value();
    parser->AddGlobalParam(name, value);
    return true;
)

IMPL_VISITOR(ETag::_integer,
    parser->ReplaceDefaultValue(&node);
    std::string name = node.attribute("name").value();
    std::string value = node.attribute("value").value();
    std::cout << name << " " << value << "\n";
    return true;
)

IMPL_VISITOR(ETag::_float,
    parser->ReplaceDefaultValue(&node);
    std::string name = node.attribute("name").value();
    std::string value = node.attribute("value").value();
    std::cout << name << " " << value << "\n";
    return true;
)

IMPL_VISITOR(ETag::_string,
    parser->ReplaceDefaultValue(&node);
    std::string name = node.attribute("name").value();
    std::string value = node.attribute("value").value();
    std::cout << name << " " << value << "\n";
    return true;
)

IMPL_VISITOR(ETag::_integrator,
    parser->ReplaceDefaultValue(&node);
    return true;
)
//IMPL_VISITOR(ETag::_sensor,)
//IMPL_VISITOR(ETag::_transform,)
//IMPL_VISITOR(ETag::_film,)
//IMPL_VISITOR(ETag::_bsdf,)
//IMPL_VISITOR(ETag::_texture,)
//IMPL_VISITOR(ETag::_rgb,)
//IMPL_VISITOR(ETag::_matrix,)
//IMPL_VISITOR(ETag::_boolean,)
//IMPL_VISITOR(ETag::_ref,)

// clang-format on

using VisitorFunc = std::function<bool(Parser *, pugi::xml_node &)>;

#define TAG_VISITOR(tag) Visitor<ETag::##_##tag>()
#define TAG_VISITORS_DEFINE(...)                           \
    std::array<VisitorFunc, 1 + TAG_ARGS_NUM(__VA_ARGS__)> \
        S_TAG_VISITORS = { Visitor<ETag::UNKNOWN>(), MAP_LIST(TAG_VISITOR, __VA_ARGS__) };

TAG_VISITORS_DEFINE(PUPIL_XML_TAGS);

[[nodiscard]] bool Visit(ETag tag, Parser *parser, pugi::xml_node &node) {
    bool flag = true;
    switch (tag) {
        case scene::xml::ETag::UNKNOWN:
        case scene::xml::ETag::COUNT:
            flag = S_TAG_VISITORS[0](parser, node);
            break;
        default:
            flag = S_TAG_VISITORS[static_cast<unsigned int>(tag)](parser, node);
            break;
    }
    return flag;
}
}// namespace scene::xml