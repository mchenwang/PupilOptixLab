#include "loader.h"

#include "pugixml.hpp"

#include <iostream>
#include <format>

using namespace scene;

void PrintNode(pugi::xml_node node, std::string pre = "") {
    std::cout << std::format("{}{}: ", pre, node.name());
    for (pugi::xml_attribute& a : node.attributes()) {
        std::cout << std::format("[{}][{}]  ", a.name(), a.value());
    }
    std::cout << std::endl;
    for (pugi::xml_node& ch : node.children()) {
        PrintNode(ch, pre + "    ");
    }
}

void scene::LoadFromXML(std::string_view path) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.data());
    if (!result)
        return ;

    PrintNode(doc);
}