#include "shape.h"
#include "cuda/util.h"
#include "cuda/stream.h"

#include "util/id.h"

#include "mesh/mesh.h"

#include <filesystem>
#include <mutex>

#include <optix_types.h>

namespace Pupil::resource {
    const unsigned int Shape::s_input_flag = OPTIX_GEOMETRY_FLAG_NONE;

    Shape::Shape(std::string_view name) noexcept
        : Object(name), m_data_dirty(true) {
        m_upload_event = std::make_unique<cuda::Event>();
    }

    Shape::~Shape() noexcept {
        m_upload_event.reset();
    }

    void Shape::WaitForDataUploading() noexcept {
        if (m_upload_event) {
            m_upload_event->Synchronize();
        }
    }

    struct ShapeManager::Impl {
        // std::mutex mutex; TODO: thread safe

        // use absolute path to identify shape
        std::unordered_map<std::string, uint64_t, util::StringHash, std::equal_to<>> map_path_to_id;
        // allow the same name
        std::unordered_multimap<std::string, uint64_t, util::StringHash, std::equal_to<>> map_name_to_id;
        std::unordered_map<uint64_t, util::Data<Shape>>                                   map_shape;
        std::unordered_map<uint64_t, std::string>                                         map_mesh_id_to_path;

        util::Data<Cube> default_shape = nullptr;

        util::UintIdAllocator id_allocation;

        Shape* GetShape(uint64_t id) noexcept {
            if (auto it = map_shape.find(id); it != map_shape.end())
                return it->second.Get();
            return nullptr;
        }
    };

    ShapeManager::ShapeManager() noexcept {
        if (m_impl) return;
        m_impl = new Impl();

        m_impl->default_shape = std::make_unique<Cube>(Shape::UserDisableTag{}, DEFAULT_SHAPE_NAME);
    }

    ShapeManager::~ShapeManager() noexcept {}

    void ShapeManager::SetShapeName(uint64_t id, std::string_view name) noexcept {
        auto shape = m_impl->GetShape(id);
        if (shape == nullptr || shape->GetName() == name) return;
        auto range = m_impl->map_name_to_id.equal_range(shape->GetName());
        auto it    = range.first;
        for (; it != range.second; ++it) {
            if (it->second == id) break;
        }
        if (it != range.second)
            m_impl->map_name_to_id.erase(it);

        if (!shape->m_name.empty())
            Log::Info("shape rename {} to {}.", shape->m_name, name);

        shape->m_name = name;
        m_impl->map_name_to_id.emplace(name, id);
    }

    util::CountableRef<Shape> ShapeManager::Register(util::Data<Shape>&& shape) noexcept {
        auto id     = m_impl->id_allocation.Allocate();
        auto name   = shape->GetName();
        shape->m_id = id;

        auto ref = shape.GetRef();
        m_impl->map_shape.emplace(id, std::move(shape));

        m_impl->map_name_to_id.emplace(name, id);
        return ref;
    }

    util::CountableRef<Shape> ShapeManager::Clone(const util::CountableRef<Shape>& shape) noexcept {
        return Register(util::Data<Shape>(shape->Clone()));
    }

    util::CountableRef<Shape> ShapeManager::LoadShapeFromFile(std::string_view path, EShapeLoadFlag flags, std::string_view name) noexcept {
        if (auto it = m_impl->map_path_to_id.find(path); it != m_impl->map_path_to_id.end()) {
            Log::Info("shape reuse [{}].", path);
            return m_impl->map_shape.at(it->second).GetRef();
        }

        auto file_path = std::filesystem::path(path);
        if (!file_path.has_extension()) {
            Log::Warn("mesh file [{}] needs to have a extension.", path);
            Log::Warn("the mesh will be replaced by default shape.");
            return m_impl->default_shape.GetRef();
        }

        std::string shape_name = name.empty() ? file_path.stem().string() : std::string{name};
        auto        extension  = file_path.extension();
        if (extension == ".obj") {
            if (ObjMesh mesh; ObjMesh::Load(path.data(), mesh, flags)) {
                auto shape = std::make_unique<TriangleMesh>(Shape::UserDisableTag{}, shape_name);
                shape->SetVertex(mesh.vertex.data(), mesh.vertex.size() / 3);
                shape->SetNormal(mesh.normal.data(), mesh.normal.size() / 3);
                shape->SetTexcoord(mesh.texcoord.data(), mesh.texcoord.size() / 2);
                shape->SetIndex(mesh.index.data(), mesh.index.size() / 3);

                auto ref = Register(util::Data<Shape>(std::move(shape)));

                m_impl->map_path_to_id[std::string{path}] = ref->GetId();
                return ref;
            }
        } else if (extension == ".hair") {
            if (CyHair hair; CyHair::Load(path.data(), hair)) {
                auto shape = std::make_unique<CurveHair>(Shape::UserDisableTag{}, Curve::EType::Cubic, shape_name);
                shape->SetCtrlVertex(hair.positions.data(), hair.positions.size() / 3);
                shape->SetWidth(hair.widths.data(), hair.widths.size(), false);
                shape->SetStrandHeadCtrlVertexIndex(hair.strands_index.data(), hair.strands_index.size());

                auto ref = Register(util::Data<Shape>(std::move(shape)));

                m_impl->map_path_to_id[std::string{path}] = ref->GetId();
                return ref;
            }
        } else {
            Log::Warn("mesh format [{}] does not support.", extension.string());
        }

        Log::Warn("shape load failed. [{}] will be replaced by default shape.", path);
        return m_impl->default_shape.GetRef();
    }

    std::vector<const Shape*> ShapeManager::GetShape(std::string_view name) noexcept {
        std::vector<const Shape*> shapes;

        if (name == DEFAULT_SHAPE_NAME) {
            shapes.push_back(m_impl->default_shape.Get());
        } else {
            auto range = m_impl->map_name_to_id.equal_range(name);
            for (auto it = range.first; it != range.second; ++it) {
                shapes.push_back(m_impl->map_shape.at(it->second).Get());
            }
        }

        return shapes;
    }

    util::CountableRef<Shape> ShapeManager::GetShape(uint64_t id) noexcept {
        if (auto it = m_impl->map_shape.find(id); it != m_impl->map_shape.end())
            return it->second.GetRef();

        Log::Warn("shape [{}] does not exist.", id);
        return m_impl->default_shape.GetRef();
    }

    std::vector<const Shape*> ShapeManager::GetShapes() const noexcept {
        std::vector<const Shape*> shapes;
        shapes.reserve(m_impl->map_shape.size());
        for (auto&& [id, shape] : m_impl->map_shape)
            shapes.push_back(shape.Get());
        return shapes;
    }

    void ShapeManager::Clear() noexcept {
        for (auto it = m_impl->map_shape.begin(); it != m_impl->map_shape.end();) {
            if (it->second.GetRefCount() == 0) {
                auto id = it->second->GetId();
                if (auto path = m_impl->map_mesh_id_to_path.find(id);
                    path != m_impl->map_mesh_id_to_path.end()) {
                    m_impl->map_path_to_id.erase(path->second);
                    m_impl->map_mesh_id_to_path.erase(path);
                }

                auto range      = m_impl->map_name_to_id.equal_range(it->second->GetName());
                auto name_id_it = range.first;
                for (; name_id_it != range.second; ++name_id_it) {
                    if (name_id_it->second == id) break;
                }
                if (name_id_it != range.second)
                    m_impl->map_name_to_id.erase(name_id_it);

                m_impl->id_allocation.Recycle(id);

                it = m_impl->map_shape.erase(it);
            } else
                ++it;
        }
    }
}// namespace Pupil::resource