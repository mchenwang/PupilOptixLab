#pragma once

#include "object.h"
#include "texture.h"

namespace Pupil::optix {
    struct Material;
}

namespace Pupil::resource {
    class Material : public Object {
    public:
        Material(std::string_view name = "") noexcept;
        virtual ~Material() noexcept;

        virtual void             UploadToCuda() noexcept     = 0;
        virtual optix::Material  GetOptixMaterial() noexcept = 0;
        virtual std::string_view GetResourceType() const noexcept override { return "Material"; }

        uint64_t GetId() const noexcept { return m_id; }

    protected:
        friend class MaterialManager;
        struct UserDisableTag {
            explicit UserDisableTag() = default;
        };

        uint64_t m_id;
    };

    class MaterialManager final : public util::Singleton<MaterialManager> {
    public:
        MaterialManager() noexcept;
        ~MaterialManager() noexcept;

        static constexpr std::string_view DEFAULT_MATERIAL_NAME = "Default Material";

        util::CountableRef<Material> Register(util::Data<Material>&& material) noexcept;
        util::CountableRef<Material> Clone(const util::CountableRef<Material>& material) noexcept;

        void SetMaterialName(uint64_t id, std::string_view name) noexcept;

        std::vector<const Material*> GetMaterial(std::string_view name) noexcept;
        util::CountableRef<Material> GetMaterial(uint64_t id) noexcept;

        std::vector<const Material*> GetMaterials() const noexcept;

        void Clear() noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };

    class Twosided : public Material {
    public:
        Twosided(UserDisableTag, std::string_view name = "") noexcept;
        ~Twosided() noexcept;

        static util::CountableRef<Material> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Material GetOptixMaterial() noexcept override;

        void SetMaterial(const util::CountableRef<Material>& material) noexcept { m_inner_material = material; }
        auto GetMaterial() const noexcept { return m_inner_material; }

    private:
        virtual void* Clone() const noexcept override;

        util::CountableRef<Material> m_inner_material;
    };

    class Diffuse : public Material {
    public:
        Diffuse(UserDisableTag, std::string_view name = "", const util::Float3& c = util::Float3(0.f)) noexcept;
        ~Diffuse() noexcept;

        static util::CountableRef<Material> Make(std::string_view name = "") noexcept;
        static util::CountableRef<Material> Make(const util::Float3& c, std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Material GetOptixMaterial() noexcept override;

        void SetReflectance(const util::Float3& color) noexcept;
        void SetReflectance(const TextureInstance& reflectance) noexcept;

        TextureInstance GetReflectance() const noexcept { return m_reflectance; }

    private:
        virtual void* Clone() const noexcept override;

        TextureInstance m_reflectance;
    };

    class Dielectric final : public Material {
    public:
        Dielectric(UserDisableTag, std::string_view name = "") noexcept;
        ~Dielectric() noexcept;

        static util::CountableRef<Material> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Material GetOptixMaterial() noexcept override;

        void SetIntIOR(float ior) noexcept;
        void SetExtIOR(float ior) noexcept;
        void SetSpecularReflectance(const util::Float3& reflectance) noexcept;
        void SetSpecularTransmittance(const util::Float3& transmittance) noexcept;
        void SetSpecularReflectance(const TextureInstance& reflectance) noexcept;
        void SetSpecularTransmittance(const TextureInstance& transmittance) noexcept;

        auto GetIntIOR() const noexcept { return m_int_ior; }
        auto GetExtIOR() const noexcept { return m_ext_ior; }
        auto GetSpecularReflectance() const noexcept { return m_specular_reflectance; }
        auto GetSpecularTransmittance() const noexcept { return m_specular_transmittance; }

    private:
        virtual void* Clone() const noexcept override;

        float           m_int_ior;
        float           m_ext_ior;
        TextureInstance m_specular_reflectance;
        TextureInstance m_specular_transmittance;
    };

    class RoughDielectric final : public Material {
    public:
        RoughDielectric(UserDisableTag, std::string_view name = "") noexcept;
        ~RoughDielectric() noexcept;

        static util::CountableRef<Material> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Material GetOptixMaterial() noexcept override;

        void SetIntIOR(float ior) noexcept;
        void SetExtIOR(float ior) noexcept;
        void SetAlpha(const util::Float3& alpha) noexcept;
        void SetSpecularReflectance(const util::Float3& reflectance) noexcept;
        void SetSpecularTransmittance(const util::Float3& transmittance) noexcept;
        void SetAlpha(const TextureInstance& alpha) noexcept;
        void SetSpecularReflectance(const TextureInstance& reflectance) noexcept;
        void SetSpecularTransmittance(const TextureInstance& transmittance) noexcept;

        auto GetIntIOR() const noexcept { return m_int_ior; }
        auto GetExtIOR() const noexcept { return m_ext_ior; }
        auto GetAlpha() const noexcept { return m_alpha; }
        auto GetSpecularReflectance() const noexcept { return m_specular_reflectance; }
        auto GetSpecularTransmittance() const noexcept { return m_specular_transmittance; }

    private:
        virtual void* Clone() const noexcept override;

        float           m_int_ior;
        float           m_ext_ior;
        TextureInstance m_alpha;
        TextureInstance m_specular_reflectance;
        TextureInstance m_specular_transmittance;
    };

    class Conductor : public Material {
    public:
        Conductor(UserDisableTag, std::string_view name = "") noexcept;
        ~Conductor() noexcept;

        static util::CountableRef<Material> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Material GetOptixMaterial() noexcept override;

        void SetEta(const util::Float3& eta) noexcept;
        void SetK(const util::Float3& k) noexcept;
        void SetSpecularReflectance(const util::Float3& reflectance) noexcept;
        void SetEta(const TextureInstance& eta) noexcept;
        void SetK(const TextureInstance& k) noexcept;
        void SetSpecularReflectance(const TextureInstance& reflectance) noexcept;

        auto GetEta() const noexcept { return m_eta; }
        auto GetK() const noexcept { return m_k; }
        auto GetSpecularReflectance() const noexcept { return m_specular_reflectance; }

    private:
        virtual void* Clone() const noexcept override;

        TextureInstance m_eta;
        TextureInstance m_k;
        TextureInstance m_specular_reflectance;
    };

    class RoughConductor : public Material {
    public:
        RoughConductor(UserDisableTag, std::string_view name = "") noexcept;
        ~RoughConductor() noexcept;

        static util::CountableRef<Material> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Material GetOptixMaterial() noexcept override;

        void SetAlpha(const util::Float3& alpha) noexcept;
        void SetEta(const util::Float3& eta) noexcept;
        void SetK(const util::Float3& k) noexcept;
        void SetSpecularReflectance(const util::Float3& reflectance) noexcept;
        void SetAlpha(const TextureInstance& alpha) noexcept;
        void SetEta(const TextureInstance& eta) noexcept;
        void SetK(const TextureInstance& k) noexcept;
        void SetSpecularReflectance(const TextureInstance& reflectance) noexcept;

        auto GetAlpha() const noexcept { return m_alpha; }
        auto GetEta() const noexcept { return m_eta; }
        auto GetK() const noexcept { return m_k; }
        auto GetSpecularReflectance() const noexcept { return m_specular_reflectance; }

    private:
        virtual void* Clone() const noexcept override;

        TextureInstance m_alpha;
        TextureInstance m_eta;
        TextureInstance m_k;
        TextureInstance m_specular_reflectance;
    };

    class Plastic : public Material {
    public:
        Plastic(UserDisableTag, std::string_view name = "") noexcept;
        ~Plastic() noexcept;

        static util::CountableRef<Material> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Material GetOptixMaterial() noexcept override;

        void SetIntIOR(float ior) noexcept;
        void SetExtIOR(float ior) noexcept;
        void SetLinear(float is_linear) noexcept;
        void SetDiffuseReflectance(const util::Float3& diffuse) noexcept;
        void SetSpecularReflectance(const util::Float3& specular) noexcept;
        void SetDiffuseReflectance(const TextureInstance& diffuse) noexcept;
        void SetSpecularReflectance(const TextureInstance& specular) noexcept;

        auto GetIntIOR() const noexcept { return m_int_ior; }
        auto GetExtIOR() const noexcept { return m_ext_ior; }
        auto GetLinear() const noexcept { return m_nonlinear; }
        auto GetDiffuseReflectance() const noexcept { return m_diffuse_reflectance; }
        auto GetSpecularReflectance() const noexcept { return m_specular_reflectance; }

    private:
        virtual void* Clone() const noexcept override;

        float           m_int_ior;
        float           m_ext_ior;
        bool            m_nonlinear;
        TextureInstance m_diffuse_reflectance;
        TextureInstance m_specular_reflectance;
    };

    class RoughPlastic : public Material {
    public:
        RoughPlastic(UserDisableTag, std::string_view name = "") noexcept;
        ~RoughPlastic() noexcept;

        static util::CountableRef<Material> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Material GetOptixMaterial() noexcept override;

        void SetIntIOR(float ior) noexcept;
        void SetExtIOR(float ior) noexcept;
        void SetLinear(float is_linear) noexcept;
        void SetAlpha(const util::Float3& alpha) noexcept;
        void SetDiffuseReflectance(const util::Float3& diffuse) noexcept;
        void SetSpecularReflectance(const util::Float3& specular) noexcept;
        void SetAlpha(const TextureInstance& alpha) noexcept;
        void SetDiffuseReflectance(const TextureInstance& diffuse) noexcept;
        void SetSpecularReflectance(const TextureInstance& specular) noexcept;

        auto GetIntIOR() const noexcept { return m_int_ior; }
        auto GetExtIOR() const noexcept { return m_ext_ior; }
        auto GetLinear() const noexcept { return m_nonlinear; }
        auto GetAlpha() const noexcept { return m_alpha; }
        auto GetDiffuseReflectance() const noexcept { return m_diffuse_reflectance; }
        auto GetSpecularReflectance() const noexcept { return m_specular_reflectance; }

    private:
        virtual void* Clone() const noexcept override;

        float           m_int_ior;
        float           m_ext_ior;
        bool            m_nonlinear;
        TextureInstance m_alpha;
        TextureInstance m_diffuse_reflectance;
        TextureInstance m_specular_reflectance;
    };

    class Hair : public Material {
    public:
        Hair(UserDisableTag, std::string_view name = "") noexcept;
        ~Hair() noexcept;

        static util::CountableRef<Material> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Material GetOptixMaterial() noexcept override;

        void SetBetaM(float beta_m) noexcept;
        void SetBetaN(float beta_n) noexcept;
        void SetAlpha(float alpha) noexcept;
        void SetSigmaA(const util::Float3& sigma_a) noexcept;
        void SetSigmaA(const TextureInstance& sigma_a) noexcept;

        auto GetBetaM() const noexcept { return m_beta_m; }
        auto GetBetaN() const noexcept { return m_beta_n; }
        auto GetAlpha() const noexcept { return m_alpha; }
        auto GetSigmaA() const noexcept { return m_sigma_a; }

    private:
        virtual void* Clone() const noexcept override;

        float           m_beta_m;
        float           m_beta_n;
        float           m_alpha;
        TextureInstance m_sigma_a;
    };
}// namespace Pupil::resource