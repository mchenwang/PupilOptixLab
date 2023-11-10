#pragma once
#include "object.h"

#include "util/util.h"
#include "util/aabb.h"
#include "util/data.h"

#include <cuda.h>
#include <string>
#include <memory>

struct OptixBuildInput;
namespace Pupil {
    namespace optix {
        struct Geometry;
    }
    namespace cuda {
        class Stream;
        class Event;
    }// namespace cuda
}// namespace Pupil

namespace Pupil::resource {
    enum class EShapeLoadFlag : unsigned int {
        None             = 0,
        GenNormals       = 1,
        GenSmoothNormals = 1 << 1,
        GenUVCoords      = 1 << 2,
        FilpUV           = 1 << 3,
        GenTanget        = 1 << 4,
        OptimizeMesh     = 1 << 5
    };

    inline bool operator&(EShapeLoadFlag target, EShapeLoadFlag type) {
        return static_cast<unsigned int>(target) & static_cast<unsigned int>(type);
    }

    inline EShapeLoadFlag operator|(EShapeLoadFlag lhs, EShapeLoadFlag rhs) {
        return static_cast<EShapeLoadFlag>(
            static_cast<unsigned int>(lhs) | static_cast<unsigned int>(rhs));
    }

    class Shape : public Object {
    public:
        util::AABB                aabb;
        static const unsigned int s_input_flag;

        Shape(std::string_view name = "") noexcept;
        virtual ~Shape() noexcept;

        virtual std::string_view GetResourceType() const noexcept override { return "Shape"; }

        virtual util::AABB GetAABB() const noexcept { return aabb; }

        virtual void            UploadToCuda() noexcept       = 0;
        virtual optix::Geometry GetOptixGeometry() noexcept   = 0;
        virtual OptixBuildInput GetOptixBuildInput() noexcept = 0;
        // TODO: virtual void OnImGui() noexcept = 0;

        void WaitForDataUploading() noexcept;

        uint64_t GetId() const noexcept { return m_id; }

    protected:
        friend class ShapeManager;
        struct UserDisableTag {
            explicit UserDisableTag() = default;
        };

        bool     m_data_dirty;
        uint64_t m_id;

        std::unique_ptr<cuda::Event> m_upload_event;
    };

    /**
     * memory manager for shape
     * @note a shape in heap memory can be registered by calling Register()
     * @note each managed shape has a unique id
     * @note unused shape memory can be cleaned up by calling the Clear()
    */
    class ShapeManager final : public util::Singleton<ShapeManager> {
    public:
        ShapeManager() noexcept;
        ~ShapeManager() noexcept;

        static constexpr std::string_view DEFAULT_SHAPE_NAME = "Default Shape";

        /** 
         * heap memory will be automatically managed if register it to the manager
         * @return a countable reference of the shape
        */
        util::CountableRef<Shape> Register(util::Data<Shape>&& shape) noexcept;
        util::CountableRef<Shape> Clone(const util::CountableRef<Shape>& shape) noexcept;

        void SetShapeName(uint64_t id, std::string_view name) noexcept;

        /**
         * load a mesh or hair shape from file
         * @param path the absolute path of asset file
        */
        util::CountableRef<Shape> LoadShapeFromFile(std::string_view path, EShapeLoadFlag flags, std::string_view name = "") noexcept;

        // get shape by name
        std::vector<const Shape*> GetShape(std::string_view name) noexcept;
        // get shape by id
        util::CountableRef<Shape> GetShape(uint64_t id) noexcept;

        /** 
         * view the shapes in memory
         * @return all registered shapes' pointer
        */
        std::vector<const Shape*> GetShapes() const noexcept;

        /**
         * clear the memory not referenced externally
        */
        void Clear() noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };

    /**
     * @note Using just one sphere data and representing different radius or center 
     * by transformation is the better choice.
    */
    class Sphere final : public Shape {
    public:
        Sphere(UserDisableTag, std::string_view name = "") noexcept;
        Sphere(UserDisableTag, const util::Float3& c, float r, std::string_view name = "") noexcept;
        ~Sphere() noexcept;

        static util::CountableRef<Shape> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Geometry GetOptixGeometry() noexcept override;
        virtual OptixBuildInput GetOptixBuildInput() noexcept override;

        void SetCenter(const util::Float3& center) noexcept;
        void SetRadius(float radius) noexcept;
        void SetFlipNormal(bool flip_normal) noexcept;

        const auto GetCenter() const noexcept { return m_center; }
        const auto GetRadius() const noexcept { return m_radius; }
        const auto GetFlipNormal() const noexcept { return m_flip_normal; }

    private:
        virtual void* Clone() const noexcept override;

        bool         m_flip_normal;
        float        m_radius;
        util::Float3 m_center;

        CUdeviceptr m_device_memory_radius;
        CUdeviceptr m_device_memory_center;
        CUdeviceptr m_device_memory_sbt_index;
    };

    class TriangleMesh : public Shape {
    public:
        TriangleMesh(UserDisableTag, std::string_view name = "") noexcept;
        virtual ~TriangleMesh() noexcept;

        static util::CountableRef<Shape> Make(std::string_view name = "") noexcept;
        static util::CountableRef<Shape> Make(std::string_view path, EShapeLoadFlag flags, std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual optix::Geometry GetOptixGeometry() noexcept override;
        virtual OptixBuildInput GetOptixBuildInput() noexcept override;

        void SetVertex(const float* vertex, uint32_t num_vertex) noexcept;
        void SetIndex(const uint32_t* index, uint32_t num_face) noexcept;
        void SetNormal(const float* normal, uint32_t num_vertex) noexcept;
        void SetTexcoord(const float* texcoord, uint32_t num_vertex) noexcept;
        void SetFlipNormal(bool flip_normal) noexcept;
        void SetFlipTexcoord(bool flip_texcoord) noexcept;

        const auto  GetFaceNum() const noexcept { return m_num_face; }
        const auto  GetVertexNum() const noexcept { return m_num_vertex; }
        const auto  GetFlipNormal() const noexcept { return m_flip_normal; }
        const auto  GetFlipTexcoord() const noexcept { return m_flip_texcoord; }
        const auto* GetVertex() const noexcept { return m_vertex.get(); }
        const auto* GetNormal() const noexcept { return m_normal.get(); }
        const auto* GetTexcoord() const noexcept { return m_texcoord.get(); }
        const auto* GetIndex() const noexcept { return m_index.get(); }

    protected:
        virtual void* Clone() const noexcept override;

        bool m_flip_normal;
        bool m_flip_texcoord;

        uint32_t                    m_num_vertex;
        std::unique_ptr<float[]>    m_vertex;
        std::unique_ptr<float[]>    m_normal;
        std::unique_ptr<float[]>    m_texcoord;
        uint32_t                    m_num_face;
        std::unique_ptr<uint32_t[]> m_index;

        CUdeviceptr m_device_memory_vertex;
        CUdeviceptr m_device_memory_normal;
        CUdeviceptr m_device_memory_texcoord;
        CUdeviceptr m_device_memory_index;
        CUdeviceptr m_device_memory_sbt_index;
    };

    class Curve : public Shape {
    public:
        enum class EType : uint32_t {
            Quadratic,
            Cubic,
            Linear,
            Catrom
        };

        Curve(UserDisableTag, EType type = EType::Cubic, std::string_view name = "") noexcept;
        virtual ~Curve() noexcept;

        static util::CountableRef<Shape> Make(EType type, std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

        virtual void            UploadToCuda() noexcept override;
        virtual util::AABB      GetAABB() const noexcept override;
        virtual optix::Geometry GetOptixGeometry() noexcept override;
        virtual OptixBuildInput GetOptixBuildInput() noexcept override;

        void SetCurveType(EType type) noexcept;
        void SetCtrlVertex(const float* ctrl_vertex, uint32_t num_ctrl_vertex) noexcept;
        void SetStrandHeadCtrlVertexIndex(const uint32_t* strand_vertex_index, uint32_t num_strand) noexcept;
        void SetWidth(float width) noexcept;
        void SetWidth(const float* width, uint32_t num_width) noexcept;

        const auto  GetCurveType() const noexcept { return m_curve_type; }
        const auto  GetCtrlVertexNum() const noexcept { return m_num_ctrl_vertex; }
        const auto  GetStrandNum() const noexcept { return m_num_strand; }
        const auto  GetSegmentCtrlVertexIndexNum() const noexcept { return m_num_segment_vertex_index; }
        const auto* GetCtrlVertex() const noexcept { return m_ctrl_vertex.get(); }
        const auto* GetWidth() const noexcept { return m_width.get(); }
        const auto* GetSegmentCtrlVertexIndex() const noexcept { return m_segment_ctrl_vertex_index.get(); }
        const auto* GetStrandHeadCtrlVertexIndex() const noexcept { return m_strand_head_ctrl_vertex_index.get(); }

        auto GetDegree() const noexcept {
            switch (m_curve_type) {
                case EType::Linear: return 1;
                case EType::Quadratic: return 2;
                case EType::Cubic: return 3;
                case EType::Catrom: return 3;
            }
            return 0;
        }

    protected:
        virtual void* Clone() const noexcept override;

        EType                       m_curve_type;
        uint32_t                    m_num_ctrl_vertex;
        uint32_t                    m_num_segment_vertex_index;
        uint32_t                    m_num_strand;
        float                       m_max_width;
        std::unique_ptr<float[]>    m_ctrl_vertex;                  //size: m_num_ctrl_vertex
        std::unique_ptr<float[]>    m_width;                        //size: m_num_ctrl_vertex
        std::unique_ptr<uint32_t[]> m_segment_ctrl_vertex_index;    //size: m_num_segment_vertex_index
        std::unique_ptr<uint32_t[]> m_strand_head_ctrl_vertex_index;//size: m_num_strand

        CUdeviceptr m_device_memory_ctrl_vertex;
        CUdeviceptr m_device_memory_width;
        CUdeviceptr m_device_memory_segment_ctrl_vertex_index;
    };

    class Rectangle final : public TriangleMesh {
    public:
        Rectangle(UserDisableTag, std::string_view name = "") noexcept;
        virtual ~Rectangle() noexcept = default;

        static util::CountableRef<Shape> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

    private:
        virtual void* Clone() const noexcept override;
    };

    class Cube final : public TriangleMesh {
    public:
        Cube(UserDisableTag, std::string_view name = "") noexcept;
        virtual ~Cube() noexcept = default;

        static util::CountableRef<Shape> Make(std::string_view name = "") noexcept;

        virtual uint64_t GetMemorySizeInByte() const noexcept override;

    private:
        virtual void* Clone() const noexcept override;
    };

    class CurveHair final : public Curve {
    public:
        CurveHair(UserDisableTag, Curve::EType type = Curve::EType::Cubic, std::string_view name = "") noexcept;
        virtual ~CurveHair() noexcept = default;

        static util::CountableRef<Shape> Make(EType type, std::string_view name = "") noexcept;
        static util::CountableRef<Shape> Make(std::string_view path, std::string_view name = "") noexcept;

        void SetWidthStyle(bool tapered) noexcept;
        void SetWidth(float width, bool tapered) noexcept;
        void SetWidth(const float* width, uint32_t num_width, bool tapered) noexcept;

        bool IsTapered() const noexcept { return m_tapered_width; }

    private:
        virtual void* Clone() const noexcept override;

        bool m_tapered_width;
    };
}// namespace Pupil::resource