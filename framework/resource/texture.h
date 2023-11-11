#pragma once
#include "object.h"

#include "util/math.h"
#include "util/util.h"
#include "util/data.h"

#include <utility>
#include <cuda_runtime.h>

namespace Pupil::optix {
    struct Texture;
}

namespace Pupil::resource {
    class Texture : public Object {
    public:
        enum class EAddressMode : unsigned int {
            Wrap   = 0,
            Clamp  = 1,
            Mirror = 2,
            Border = 3
        };

        enum class EFilterMode : unsigned int {
            Point  = 0,
            Linear = 1
        };

        enum class EColorSpace {
            sRGB,
            Linear
        };

        Texture(std::string_view name) noexcept;
        virtual ~Texture() noexcept;

        Texture(const Texture&)            = delete;
        Texture& operator=(const Texture&) = delete;

        virtual std::string_view GetResourceType() const noexcept override { return "Texture"; }

        virtual void           UploadToCuda() noexcept {}
        virtual Float3         GetPixelAverage() const noexcept = 0;
        virtual optix::Texture GetOptixTexture() noexcept       = 0;
        // TODO: virtual void OnImGui() noexcept;

        uint64_t GetId() const noexcept { return m_id; }

    protected:
        friend class TextureManager;
        struct UserDisableTag {
            explicit UserDisableTag() = default;
        };

        uint64_t m_id;
    };

    class TextureInstance {
    public:
        TextureInstance()  = default;
        ~TextureInstance() = default;

        TextureInstance(const util::CountableRef<Texture>& texture) noexcept { m_resource = texture; }
        TextureInstance& operator=(const util::CountableRef<Texture>& texture) noexcept {
            m_resource = texture;
            return *this;
        }

        TextureInstance(const TextureInstance& other) noexcept {
            m_resource  = other.m_resource;
            m_transform = other.m_transform;
        }
        TextureInstance& operator=(const TextureInstance& other) noexcept {
            m_resource  = other.m_resource;
            m_transform = other.m_transform;
            return *this;
        }

        auto operator->() const noexcept { return m_resource; }

        operator util::CountableRef<Texture>() const noexcept { return m_resource; }
        operator bool() const noexcept { return m_resource.Get() != nullptr; }

        void SetTexture(const util::CountableRef<Texture>& texture) { m_resource = texture; }
        void SetTransform(const Transform& transform) noexcept { m_transform = transform; }

        auto& GetTexture() noexcept { return m_resource; }
        auto  GetTransform() const noexcept { return m_transform; }

        optix::Texture GetOptixTexture() noexcept;

    private:
        util::CountableRef<Texture> m_resource;
        Transform                   m_transform;
    };

    /**
     * memory manager for texture
     * @note a texture in heap memory can be registered by calling Register()
     * @note each managed texture has a unique id
     * @note unused texture memory can be cleaned up by calling the Clear()
    */
    class TextureManager final : public util::Singleton<TextureManager> {
    public:
        TextureManager() noexcept;
        ~TextureManager() noexcept;

        static constexpr std::string_view DEFAULT_TEXTURE_NAME = "Default Texture";

        /** 
         * heap memory will be automatically managed if register it to the manager
         * @return a countable reference of the texture
        */
        util::CountableRef<Texture> Register(util::Data<Texture>&& texture) noexcept;
        util::CountableRef<Texture> Clone(const util::CountableRef<Texture>& texture) noexcept;

        void SetTextureName(uint64_t id, std::string_view name) noexcept;

        /**
         * load image from file
         * @param path the absolute path of image
         * @param sRGB if sRGB is true, the data will be raised to the power of 2.2, in order to convert the image into a linear space.
        */
        util::CountableRef<Texture> LoadTextureFromFile(std::string_view path, bool sRGB = true, std::string_view name = "") noexcept;

        // get texture by name
        std::vector<const Texture*> GetTexture(std::string_view name) noexcept;
        // get texture by id
        util::CountableRef<Texture> GetTexture(uint64_t id) noexcept;

        /** 
         * view the textures in memory
         * @return all registered textures' pointer
        */
        std::vector<const Texture*> GetTextures() const noexcept;

        /**
         * clear the memory not referenced externally
        */
        void Clear() noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };

    class Bitmap final : public Texture {
    public:
        Bitmap(UserDisableTag, std::string_view name) noexcept;
        Bitmap(UserDisableTag, std::string_view name, size_t w, size_t h,
               const float* data         = nullptr,
               EAddressMode address_mode = EAddressMode::Wrap,
               EFilterMode  filter_mode  = EFilterMode::Linear) noexcept;
        ~Bitmap() noexcept;

        static util::CountableRef<Texture> Make(std::string_view name = "") noexcept;
        static util::CountableRef<Texture> Make(std::string_view path, bool sRGB = true, std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void           UploadToCuda() noexcept override;
        virtual Float3         GetPixelAverage() const noexcept override;
        virtual optix::Texture GetOptixTexture() noexcept override;

        void SetAddressMode(EAddressMode address_mode) noexcept;
        void SetFilterMode(EFilterMode filter_mode) noexcept;
        void SetSize(uint32_t w, uint32_t h) noexcept;
        /**
         * @param offset insert the data into texture with a offset which is 4x(rgba) of pixel size
        */
        void SetData(const float* data, size_t size_of_float, size_t offset_of_float = 0) noexcept;

        auto        GetSize() const noexcept { return std::make_pair(m_width, m_height); }
        auto        GetFilterMode() const noexcept { return m_filter_mode; }
        auto        GetAddressMode() const noexcept { return m_address_mode; }
        const auto* GetData() const noexcept { return m_data.get(); }

    private:
        virtual void* Clone() const noexcept override;

        size_t m_width;
        size_t m_height;

        bool                     m_data_dirty;
        std::unique_ptr<float[]> m_data;

        EAddressMode m_address_mode;
        EFilterMode  m_filter_mode;

        cudaArray_t         m_cuda_data_array;
        cudaTextureObject_t m_cuda_tex_object;
    };

    class RGBTexture final : public Texture {
    public:
        RGBTexture(UserDisableTag, std::string_view name, const Float3& c) noexcept;

        static util::CountableRef<Texture> Make(const Float3& c, std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual Float3         GetPixelAverage() const noexcept override;
        virtual optix::Texture GetOptixTexture() noexcept override;

        void SetColor(const Float3& c) noexcept { m_color = c; }
        auto GetColor() const noexcept { return m_color; }

    private:
        virtual void* Clone() const noexcept override;

        Float3 m_color;
    };

    class CheckerboardTexture final : public Texture {
    public:
        CheckerboardTexture(UserDisableTag, std::string_view name, const Float3& c1, const Float3& c2) noexcept;

        static util::CountableRef<Texture> Make(std::string_view name = "") noexcept;
        static util::CountableRef<Texture> Make(const Float3& c1, const Float3& c2, std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual Float3         GetPixelAverage() const noexcept override;
        virtual optix::Texture GetOptixTexture() noexcept override;

        void SetCheckerboradColor1(const Float3& color) noexcept { m_checkerborad_color1 = color; }
        void SetCheckerboradColor2(const Float3& color) noexcept { m_checkerborad_color2 = color; }
        void SetCheckerboradColor(const Float3& c1, const Float3& c2) noexcept {
            m_checkerborad_color1 = c1;
            m_checkerborad_color2 = c2;
        }
        auto GetCheckerboradColor() const noexcept { return std::make_pair(m_checkerborad_color1, m_checkerborad_color2); }

    private:
        virtual void* Clone() const noexcept override;

        Float3 m_checkerborad_color1;
        Float3 m_checkerborad_color2;
    };
}// namespace Pupil::resource