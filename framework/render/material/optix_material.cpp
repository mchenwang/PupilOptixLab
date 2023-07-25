#include "optix_material.h"
#include "cuda/texture.h"
#include "optix/util.h"
#include "fresnel.h"

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

#define MATERIAL_LOAD_FUNC(type) Pupil::optix::material::##type LoadMaterial(const Pupil::resource::##type &mat) noexcept

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

MATERIAL_LOAD_FUNC(RoughDielectric) {
    Pupil::optix::material::RoughDielectric ret;
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    ret.eta = mat.int_ior / mat.ext_ior;
    ret.alpha = tex_mngr->GetCudaTexture(mat.alpha);
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
    ret.alpha = tex_mngr->GetCudaTexture(mat.alpha);
    ret.eta = tex_mngr->GetCudaTexture(mat.eta);
    ret.k = tex_mngr->GetCudaTexture(mat.k);
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
    return ret;
}

MATERIAL_LOAD_FUNC(Plastic) {
    Pupil::optix::material::Plastic ret;
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    ret.eta = mat.int_ior / mat.ext_ior;
    ret.nonlinear = mat.nonlinear;
    ret.diffuse_reflectance = tex_mngr->GetCudaTexture(mat.diffuse_reflectance);
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);

    float diffuse_luminance = Pupil::optix::GetLuminance(GetPixelAverage(mat.diffuse_reflectance));
    float specular_luminance = Pupil::optix::GetLuminance(GetPixelAverage(mat.specular_reflectance));
    ret.m_specular_sampling_weight = specular_luminance / (specular_luminance + diffuse_luminance);

    ret.m_int_fdr = Pupil::optix::material::fresnel::DiffuseReflectance(1.f / ret.eta);
    return ret;
}

MATERIAL_LOAD_FUNC(RoughPlastic) {
    Pupil::optix::material::RoughPlastic ret;
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    ret.eta = mat.int_ior / mat.ext_ior;
    ret.nonlinear = mat.nonlinear;
    ret.alpha = tex_mngr->GetCudaTexture(mat.alpha);
    ret.diffuse_reflectance = tex_mngr->GetCudaTexture(mat.diffuse_reflectance);
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);

    float diffuse_luminance = Pupil::optix::GetLuminance(GetPixelAverage(mat.diffuse_reflectance));
    float specular_luminance = Pupil::optix::GetLuminance(GetPixelAverage(mat.specular_reflectance));
    ret.m_specular_sampling_weight = specular_luminance / (specular_luminance + diffuse_luminance);

    ret.m_int_fdr = Pupil::optix::material::fresnel::DiffuseReflectance(1.f / ret.eta);
    return ret;
}

namespace Pupil::optix::material {
void Material::LoadMaterial(Pupil::resource::Material mat) noexcept {
    type = mat.type;
    twosided = mat.twosided;
    switch (type) {
#define PUPIL_MATERIAL_TYPE_ATTR_DEFINE(enum_type, mat_attr) \
    case EMatType::##enum_type:                              \
        mat_attr = ::LoadMaterial(mat.mat_attr);             \
        break;
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_ATTR_DEFINE
    }
}
}// namespace Pupil::optix::material