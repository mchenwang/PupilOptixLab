#include "util_loader.h"
#include "resource/scene.h"
#include "resource/texture.h"
#include "util/log.h"

namespace Pupil::resource::xml {

bool LoadInt(std::string_view param_name, std::string_view value,
             int &param, int default_value) noexcept {
    if (value.empty()) {
        param = default_value;
        return false;
    }

    param = std::stoi(value.data());
    return true;
}

bool LoadInt(const resource::xml::Object *obj, std::string_view param_name,
             int &param, int default_value) noexcept {
    auto value = obj->GetProperty(param_name);
    return LoadInt(param_name, value, param, default_value);
}

bool LoadFloat(std::string_view param_name, std::string_view value,
               float &param, float default_value) noexcept {
    if (value.empty()) {
        param = default_value;
        return false;
    }

    param = std::stof(value.data());
    return true;
}

bool LoadFloat(const resource::xml::Object *obj, std::string_view param_name,
               float &param, float default_value) noexcept {
    auto value = obj->GetProperty(param_name);
    return LoadFloat(param_name, value, param, default_value);
}

bool LoadFloat3(std::string_view param_name, std::string_view value,
                util::Float3 &param, util::Float3 default_value) noexcept {
    if (value.empty()) {
        param = default_value;
        return false;
    }
    auto xyz = util::Split(value, ",");
    if (xyz.size() == 3) {
        param.x = std::stof(xyz[0]);
        param.y = std::stof(xyz[1]);
        param.z = std::stof(xyz[2]);
    } else if (xyz.size() == 1) {
        param.x = param.y = param.z = std::stof(xyz[0]);
    } else {
        Pupil::Log::Warn("[{}] size is {}(must be 3 or 1).", param_name, xyz.size());
        return false;
    }
    return true;
}

bool LoadFloat3(const resource::xml::Object *obj, std::string_view param_name,
                util::Float3 &param, util::Float3 default_value) noexcept {
    auto value = obj->GetProperty(param_name);
    return LoadFloat3(param_name, value, param, default_value);
}

bool Load3Float(std::string_view param_name, std::string_view value,
                util::Float3 &param, util::Float3 default_value) noexcept {
    if (value.empty()) {
        param = default_value;
        return false;
    }
    auto xyz = util::Split(value, ",");
    if (xyz.size() == 3) {
        param.x = std::stof(xyz[0]);
        param.y = std::stof(xyz[1]);
        param.z = std::stof(xyz[2]);
    } else {
        Pupil::Log::Warn("[{}] size is {}(must be 3).", param_name, xyz.size());
        return false;
    }
    return true;
}

bool Load3Float(const resource::xml::Object *obj, std::string_view param_name,
                util::Float3 &param, util::Float3 default_value) noexcept {
    auto value = obj->GetProperty(param_name);
    return Load3Float(param_name, value, param, default_value);
}

bool LoadTextureOrRGB(const resource::xml::Object *obj, resource::Scene *scene, std::string_view param_name,
                      util::Texture &param, util::Float3 default_value) noexcept {
    auto [texture, rgb] = obj->GetParameter(param_name);

    if (texture == nullptr && rgb.empty()) {
        param = util::Singleton<resource::TextureManager>::instance()->GetColorTexture(default_value.r, default_value.g, default_value.b);
        return false;
    }

    if (texture == nullptr) {
        util::Float3 color;
        LoadFloat3(param_name, rgb, color, default_value);
        param = util::Singleton<resource::TextureManager>::instance()->GetColorTexture(color);
    } else {
        scene->LoadXmlObj(texture, &param);
    }
    return true;
}

bool LoadBool(const resource::xml::Object *obj, std::string_view param_name, bool &param, bool default_value) noexcept {
    std::string value = obj->GetProperty(param_name);
    if (value.empty()) {
        param = default_value;
        return false;
    }
    if (value.compare("true") == 0)
        param = true;
    else if (value.compare("false") == 0)
        param = false;
    else {
        param = default_value;
        return false;
    }
    return true;
}

bool LoadTransform3D(const resource::xml::Object *obj, util::Transform *transform) noexcept {
    std::string value = obj->GetProperty("matrix");
    if (!value.empty()) {
        auto matrix_elems = util::Split(value, " ");
        if (matrix_elems.size() == 16) {
            for (int i = 0; auto &&e : matrix_elems) {
                transform->matrix.e[i++] = std::stof(e);
            }
        } else if (matrix_elems.size() == 9) {
            for (int i = 0, j = 0; auto &&e : matrix_elems) {
                transform->matrix.e[i] = std::stof(e);
                ++i, ++j;
                if (j % 3 == 0) ++i;
            }
        } else {
            Pupil::Log::Warn("transform matrix size is {}(must be 9 or 16).", matrix_elems.size());
            for (size_t i = 0; i < matrix_elems.size() && i < 16; i++) {
                transform->matrix.e[i] = std::stof(matrix_elems[i]);
            }
        }
    } else {
        auto look_at = obj->GetUniqueSubObject("lookat");
        if (look_at) {
            util::Float3 origin{ 1.f, 0.f, 0.f };
            util::Float3 target{ 0.f, 0.f, 0.f };
            util::Float3 up{ 0.f, 1.f, 0.f };
            Load3Float(look_at, "origin", origin, { 1.f, 0.f, 0.f });
            Load3Float(look_at, "target", target, { 0.f, 0.f, 0.f });
            Load3Float(look_at, "up", up, { 0.f, 1.f, 0.f });
            transform->LookAt(origin, target, up);

            // Mitsuba 3: +X points left, +Y points up, +Z points view
            // Pupil Transform: +X points right, +Y points up, +Z points -view
            transform->matrix.re[0][0] *= -1;
            transform->matrix.re[1][0] *= -1;
            transform->matrix.re[2][0] *= -1;
            transform->matrix.re[0][2] *= -1;
            transform->matrix.re[1][2] *= -1;
            transform->matrix.re[2][2] *= -1;

            if (!obj->GetProperty("scale").empty() || obj->GetUniqueSubObject("rotate") || !obj->GetProperty("translate").empty()) {
                Pupil::Log::Warn("transform scale/rotate/translate is ignored as look_at exists.");
            }
            return true;
        }

        if (util::Float3 scale; LoadFloat3(obj, "scale", scale)) {
            transform->Scale(scale.x, scale.y, scale.z);
        }

        auto rotate_obj = obj->GetUniqueSubObject("rotate");
        if (rotate_obj) {
            if (util::Float3 axis; Load3Float(rotate_obj, "axis", axis)) {
                if (float angle; LoadFloat(rotate_obj, "angle", angle)) {
                    transform->Rotate(axis.x, axis.y, axis.z, angle);
                }
            }
        }

        if (util::Float3 translate; Load3Float(obj, "translate", translate)) {
            transform->Translate(translate.x, translate.y, translate.z);
        }
    }
    return true;
}

bool LoadTransform(const resource::xml::Object *obj, void *dst) noexcept {
    if (obj == nullptr || dst == nullptr) return false;
    if (obj->var_name.compare("to_world") == 0) {
        util::Transform *transform = static_cast<util::Transform *>(dst);
        return LoadTransform3D(obj, transform);
    } else if (obj->var_name.compare("to_uv") == 0) {
        util::Transform *transform = static_cast<util::Transform *>(dst);

        if (util::Float3 scale; LoadFloat3(obj, "scale", scale)) {
            transform->Scale(scale.x, scale.y, scale.z);
        }
        return true;
    } else {
        Pupil::Log::Warn("transform [{}] UNKNOWN.", obj->var_name);
    }
    return false;
}
}// namespace Pupil::resource::xml