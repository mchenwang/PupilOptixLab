#include "scene/scene.h"
#include <filesystem>

namespace Pupil::resource::mixml {
    struct Object;
    class MixmlSceneLoader : public Pupil::SceneLoader {
    public:
        virtual bool Load(std::filesystem::path path, Scene* scene) noexcept override;

    protected:
        virtual bool Visit(void* obj, Scene* scene) noexcept override;

        bool                         LoadBool(mixml::Object* xml_obj, std::string_view param_name, bool default_value) noexcept;
        int                          LoadInt(mixml::Object* xml_obj, std::string_view param_name, int default_value) noexcept;
        float                        LoadFloat(mixml::Object* xml_obj, std::string_view param_name, float default_value) noexcept;
        Float3                       LoadFloat3(mixml::Object* xml_obj, std::string_view param_name, Float3 default_value) noexcept;
        util::CountableRef<Shape>    LoadShape(mixml::Object* xml_obj) noexcept;
        util::CountableRef<Material> LoadMaterial(mixml::Object* xml_obj) noexcept;
        TextureInstance              LoadTexture(mixml::Object* xml_obj, bool sRGB, Float3 default_value) noexcept;
        TextureInstance              LoadTexture(mixml::Object* xml_obj, std::string_view param_name, bool sRGB, Float3 default_value) noexcept;
        Transform                    LoadTransform(mixml::Object* xml_obj) noexcept;
        std::unique_ptr<Emitter>     LoadEmitter(mixml::Object* xml_obj) noexcept;

        std::filesystem::path m_scene_root_path;
    };
}// namespace Pupil::resource::mixml