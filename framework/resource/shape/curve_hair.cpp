#include "resource/shape.h"
#include "util/util.h"

namespace Pupil::resource {
    util::CountableRef<Shape> CurveHair::Make(EType type, std::string_view name) noexcept {
        auto shape_mngr = util::Singleton<ShapeManager>::instance();
        return shape_mngr->Register(std::make_unique<CurveHair>(UserDisableTag{}, type, name));
    }

    util::CountableRef<Shape> CurveHair::Make(std::string_view path, std::string_view name) noexcept {
        auto shape_mngr = util::Singleton<ShapeManager>::instance();
        auto ref        = shape_mngr->LoadShapeFromFile(path, EShapeLoadFlag::None);
        if (!name.empty()) shape_mngr->SetShapeName(ref->GetId(), name);
        return ref;
    }

    CurveHair::CurveHair(UserDisableTag tag, Curve::EType type, std::string_view name) noexcept
        : Curve(tag, type, name) {
        m_tapered_width = false;
    }

    void CurveHair::SetWidthStyle(bool tapered) noexcept {
        if (tapered == m_tapered_width) return;

        SetWidth(m_width.get(), m_num_ctrl_vertex, tapered);

        m_data_dirty = true;
    }

    void CurveHair::SetWidth(float width, bool tapered) noexcept {
        SetWidth(&width, 1, tapered);
    }

    void CurveHair::SetWidth(const float* width, uint32_t num_width, bool tapered) noexcept {
        assert(width != nullptr);

        m_tapered_width = tapered;

        if (!m_tapered_width) {
            std::fill_n(m_width.get(), m_num_ctrl_vertex, *width);
        } else {
            for (int i = 0; i < m_num_strand - 1; ++i) {
                const uint32_t start = m_strand_head_ctrl_vertex_index[i];
                const uint32_t num   = m_strand_head_ctrl_vertex_index[i + 1] - start;
                for (uint32_t index = 0; index < num; ++index)
                    m_width[start + index] = (*width) * (num - 1 - index) / static_cast<float>(num - 1);
            }
        }

        m_data_dirty = true;
    }

    void* CurveHair::Clone() const noexcept {
        auto clone             = new CurveHair(UserDisableTag{}, m_curve_type, m_name);
        clone->m_tapered_width = m_tapered_width;
        clone->SetCtrlVertex(m_ctrl_vertex.get(), m_num_ctrl_vertex);
        clone->SetWidth(m_width.get(), m_num_ctrl_vertex, m_tapered_width);
        clone->SetStrandHeadCtrlVertexIndex(m_strand_head_ctrl_vertex_index.get(), m_num_strand);

        return clone;
    }

}// namespace Pupil::resource