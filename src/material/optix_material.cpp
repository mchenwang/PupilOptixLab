#include "optix_material.h"
#include "device/cuda_texture.h"

#define MATERIAL_LOAD_FUNC(type) optix_util::material::##type LoadMaterial(const ::material::##type &mat) noexcept

MATERIAL_LOAD_FUNC(Diffuse) {
    optix_util::material::Diffuse ret;
    auto tex_mngr = util::Singleton<device::CudaTextureManager>::instance();
    ret.reflectance = tex_mngr->GetCudaTexture(mat.reflectance);
    return ret;
}

MATERIAL_LOAD_FUNC(Dielectric) {
    optix_util::material::Dielectric ret;
    auto tex_mngr = util::Singleton<device::CudaTextureManager>::instance();
    ret.int_ior = mat.int_ior;
    ret.ext_ior = mat.ext_ior;
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
    ret.specular_transmittance = tex_mngr->GetCudaTexture(mat.specular_transmittance);
    return ret;
}

MATERIAL_LOAD_FUNC(Conductor) {
    optix_util::material::Conductor ret;
    auto tex_mngr = util::Singleton<device::CudaTextureManager>::instance();
    ret.eta = tex_mngr->GetCudaTexture(mat.eta);
    ret.k = tex_mngr->GetCudaTexture(mat.k);
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
    return ret;
}

MATERIAL_LOAD_FUNC(RoughConductor) {
    optix_util::material::RoughConductor ret;
    auto tex_mngr = util::Singleton<device::CudaTextureManager>::instance();
    ret.alpha = mat.alpha;
    ret.eta = tex_mngr->GetCudaTexture(mat.eta);
    ret.k = tex_mngr->GetCudaTexture(mat.k);
    ret.specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
    return ret;
}

namespace optix_util::material {
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

            // case new material
    }
}
}// namespace optix_util::material