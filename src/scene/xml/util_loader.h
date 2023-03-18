#pragma once

#include "object.h"
#include "common/util.h"
#include "common/texture.h"

namespace scene {
class Scene;

namespace xml {

bool LoadInt(std::string_view param_name, std::string_view value,
             int &param, int default_value = 0) noexcept;
bool LoadInt(const scene::xml::Object *obj, std::string_view param_name,
             int &param, int default_value = 0) noexcept;

bool LoadFloat(std::string_view param_name, std::string_view value,
               float &param, float default_value = 0.f) noexcept;
bool LoadFloat(const scene::xml::Object *obj, std::string_view param_name,
               float &param, float default_value = 0.f) noexcept;

bool LoadFloat3(std::string_view param_name, std::string_view value,
                util::Float3 &param, util::Float3 default_value = { 0.f }) noexcept;
bool LoadFloat3(const scene::xml::Object *obj, std::string_view param_name,
                util::Float3 &param, util::Float3 default_value = { 0.f }) noexcept;
bool Load3Float(std::string_view param_name, std::string_view value,
                util::Float3 &param, util::Float3 default_value = { 0.f }) noexcept;
bool Load3Float(const scene::xml::Object *obj, std::string_view param_name,
                util::Float3 &param, util::Float3 default_value = { 0.f }) noexcept;

bool LoadTextureOrRGB(const scene::xml::Object *obj, scene::Scene *scene, std::string_view param_name,
                      util::Texture &param, util::Float3 default_value = { 0.f }) noexcept;

bool LoadBool(const scene::xml::Object *obj, std::string_view param_name, bool &param, bool default_value = false) noexcept;

bool LoadTransform3D(const scene::xml::Object *obj, util::Transform *transform) noexcept;
bool LoadTransform(const scene::xml::Object *obj, void *dst) noexcept;
}// namespace xml
}// namespace scene