#include "optix_material.h"
#include "cuda/texture.h"
#include "optix/util.h"

namespace {
using namespace Pupil;

float3 GetPixelAverage(util::Texture texture) {
    switch (texture.type) {
        case util::ETextureType::RGB:
            return make_float3(texture.rgb.color.r, texture.rgb.color.g, texture.rgb.color.b);
            break;
        case util::ETextureType::Checkerboard: {
            float r = texture.checkerboard.patch1.r + texture.checkerboard.patch2.r;
            float g = texture.checkerboard.patch1.g + texture.checkerboard.patch2.g;
            float b = texture.checkerboard.patch1.b + texture.checkerboard.patch2.b;
            return make_float3(r, g, b) * 0.5f;
        } break;
        case util::ETextureType::Bitmap: {
            float r = 0.f;
            float g = 0.f;
            float b = 0.f;
            for (int i = 0, idx = 0; i < texture.bitmap.h; ++i) {
                for (int j = 0; j < texture.bitmap.w; ++j) {
                    r += texture.bitmap.data[idx++];
                    g += texture.bitmap.data[idx++];
                    b += texture.bitmap.data[idx++];
                    idx++;// a
                }
            }
            return make_float3(r, g, b) / (1.f * texture.bitmap.h * texture.bitmap.w);
        } break;
    }
    return make_float3(0.f);
}
}// namespace

#define MATERIAL_LOAD_FUNC(type) Pupil::optix::material::##type LoadMaterial(const ::material::##type &mat) noexcept

MATERIAL_LOAD_FUNC(Diffuse) {
    Pupil::optix::material::Diffuse ret;
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    ret.reflectance = tex_mngr->GetCudaTexture(mat.reflectance);
    return ret;
}

MATERIAL_LOAD_FUNC(Dielectric) {
    Pupil::optix::material::Dielectric ret;
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    ret.int_ior = mat.int_ior;
    ret.ext_ior = mat.ext_ior;
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
    ret.specular_transmittance = tex_mngr->GetCudaTexture(mat.specular_transmittance);
    return ret;
}

MATERIAL_LOAD_FUNC(Conductor) {
    Pupil::optix::material::Conductor ret;
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    ret.eta = tex_mngr->GetCudaTexture(mat.eta);
    ret.k = tex_mngr->GetCudaTexture(mat.k);
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
    return ret;
}

MATERIAL_LOAD_FUNC(RoughConductor) {
    Pupil::optix::material::RoughConductor ret;
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    ret.alpha = mat.alpha;
    ret.eta = tex_mngr->GetCudaTexture(mat.eta);
    ret.k = tex_mngr->GetCudaTexture(mat.k);
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
    return ret;
}

MATERIAL_LOAD_FUNC(Plastic) {
    Pupil::optix::material::Plastic ret;
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    ret.int_ior = mat.int_ior;
    ret.ext_ior = mat.ext_ior;
    ret.nonlinear = mat.nonlinear;
    ret.diffuse_reflectance = tex_mngr->GetCudaTexture(mat.diffuse_reflectance);
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);

    float diffuse_luminance = Pupil::optix::GetLuminance(GetPixelAverage(mat.diffuse_reflectance));
    float specular_luminance = Pupil::optix::GetLuminance(GetPixelAverage(mat.specular_reflectance));
    ret.m_specular_sampling_weight = specular_luminance / (specular_luminance + diffuse_luminance);
    return ret;
}

MATERIAL_LOAD_FUNC(RoughPlastic) {
    Pupil::optix::material::RoughPlastic ret;
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    ret.int_ior = mat.int_ior;
    ret.ext_ior = mat.ext_ior;
    ret.nonlinear = mat.nonlinear;
    ret.alpha = mat.alpha;
    ret.diffuse_reflectance = tex_mngr->GetCudaTexture(mat.diffuse_reflectance);
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);

    float diffuse_luminance = Pupil::optix::GetLuminance(GetPixelAverage(mat.diffuse_reflectance));
    float specular_luminance = Pupil::optix::GetLuminance(GetPixelAverage(mat.specular_reflectance));
    ret.m_specular_sampling_weight = specular_luminance / (specular_luminance + diffuse_luminance);
    return ret;
}

namespace Pupil::optix::material {
void Material::LoadMaterial(::material::Material mat) noexcept {
    type = mat.type;
    twosided = mat.twosided;
    switch (type) {
        case EMatType::_diffuse:
            diffuse = ::LoadMaterial(mat.diffuse);
            break;
        case EMatType::_dielectric:
            dielectric = ::LoadMaterial(mat.dielectric);
            break;
        case EMatType::_conductor:
            conductor = ::LoadMaterial(mat.conductor);
            break;
        case EMatType::_roughconductor:
            rough_conductor = ::LoadMaterial(mat.rough_conductor);
            break;
        case EMatType::_plastic:
            plastic = ::LoadMaterial(mat.plastic);
            break;
        case EMatType::_roughplastic:
            rough_plastic = ::LoadMaterial(mat.rough_plastic);
            break;

            // case new material
    }
}
}// namespace Pupil::optix::material