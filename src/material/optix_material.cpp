#include "optix_material.h"
#include "device/cuda_texture.h"

namespace optix_util::material {
void Diffuse::LoadMaterial(const ::material::Diffuse &mat) noexcept {
    auto tex_mngr = util::Singleton<device::CudaTextureManager>::instance();
    reflectance = tex_mngr->GetCudaTexture(mat.reflectance);
}

void Dielectric::LoadMaterial(const ::material::Dielectric &mat) noexcept {
    auto tex_mngr = util::Singleton<device::CudaTextureManager>::instance();
    int_ior = mat.int_ior;
    ext_ior = mat.ext_ior;
    specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
    specular_transmittance = tex_mngr->GetCudaTexture(mat.specular_transmittance);
}

void Conductor::LoadMaterial(const ::material::Conductor &mat) noexcept {
    auto tex_mngr = util::Singleton<device::CudaTextureManager>::instance();
    eta = tex_mngr->GetCudaTexture(mat.eta);
    k = tex_mngr->GetCudaTexture(mat.k);
    specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
}

void RoughConductor::LoadMaterial(const ::material::RoughConductor &mat) noexcept {
    auto tex_mngr = util::Singleton<device::CudaTextureManager>::instance();
    alpha = mat.alpha;
    eta = tex_mngr->GetCudaTexture(mat.eta);
    k = tex_mngr->GetCudaTexture(mat.k);
    specular_reflectance = tex_mngr->GetCudaTexture(mat.specular_reflectance);
}
}// namespace optix_util::material