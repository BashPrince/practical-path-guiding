
#pragma once
#if !defined(__MITSUBA_RENDER_BSDFPROXY_H_)
#define __MITSUBA_RENDER_BSDFPROXY_H_

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

class BSDFProxy
{
  public:
    BSDFProxy()
    : m_diffuse_weight(0.0f)
    , m_translucency_weight(0.0f)
    , m_reflection_weight(0.0f)
    , m_refraction_weight(0.0f)
    , m_reflection_roughness(0.0f)
    , m_refraction_roughness(0.0f)
    {}

    void add_diffuse_weight(const float diffuse_weight)
    {
        m_diffuse_weight += diffuse_weight;
    }

    void add_translucency_weight(const float translucency_weight)
    {
            m_translucency_weight += translucency_weight;
    }

    void add_reflection_weight(
        const float reflection_weight,
        const float roughness)
    {
        const float old_weight = m_reflection_weight;
        m_reflection_weight += reflection_weight;
        const float inv_weight = m_reflection_weight > 0.0f ? 1.0f / m_reflection_weight : 0.0f;
        m_reflection_roughness = old_weight * inv_weight * m_reflection_roughness + reflection_weight * inv_weight * roughness;
    }

    void add_refraction_weight(
        const float refraction_weight,
        const float roughness,
        const float ior)
    {
        const float old_weight = m_refraction_weight;
        m_refraction_weight += refraction_weight;
        const float inv_weight = m_refraction_weight > 0.0f ? 1.0f / m_refraction_weight : 0.0f;
        m_refraction_roughness = old_weight * inv_weight * m_refraction_roughness + refraction_weight * inv_weight * roughness;

        m_IOR = ior; // TO-DO: handle appropriately
    }

    void finish_parameterization(
        const Vector3f &outgoing,
        const Vector3f &shading_normal)
    {
        m_is_diffuse = m_diffuse_weight > 0.0f;
        m_is_translucent = m_translucency_weight > 0.0f;
        m_is_reflective = m_reflection_weight > 0.0f;
        m_is_refractive = m_refraction_weight > 0.0f;

        if (is_zero())
            return;

        // Construct lobes in world space.
        // TO-DO: Hemisphere checks, basic asserts.
        m_normal = shading_normal;
        m_reflection_lobe = reflect(outgoing, m_normal);
        m_refraction_lobe = refract(outgoing, m_normal, m_IOR);

        // Roughness correction.
        m_reflection_roughness *= 2.0f;
        const float cos_nt = std::abs(dot(m_normal, m_refraction_lobe));
        const float cos_no = std::abs(dot(m_normal, outgoing));
        const float scale_factor_refraction = (cos_nt + m_IOR * cos_no) / cos_nt;
        m_refraction_roughness *= scale_factor_refraction;
    }

    float evaluate(
        const Vector3f &incoming) const
    {
        float value = 0.0f;
        const float cos_ni = dot(m_normal, incoming);
        const float cos_negni = -cos_ni;

        if (m_is_diffuse)
        {
            value += m_diffuse_weight * std::max(cos_ni, 0.0f);
        }
        if (m_is_translucent)
        {
            value += m_translucency_weight * std::max(cos_negni, 0.0f);
        }
        if (m_is_reflective && cos_ni > 0.0f)
        {
            const float cos_refl_i = dot(m_reflection_lobe, incoming);

            if (cos_refl_i > 0.0f)
            {
                const float cos2_refl_i = cos_refl_i * cos_refl_i;
                const float sin2_refl_i = 1.0f - cos2_refl_i;
                const float alpha2 = m_reflection_roughness * m_reflection_roughness;

                float factor = cos2_refl_i + sin2_refl_i / alpha2;
                factor *= factor;
                factor *= M_PI * alpha2;
                value += factor = 0.0f ? 0.0f : m_reflection_weight / factor;
            }
        }
        if (m_is_refractive && cos_negni > 0.0f)
        {
            const float cos_refr_i = dot(m_refraction_lobe, incoming);

            if (cos_refr_i > 0.0f)
            {
                const float cos2_refr_i = cos_refr_i * cos_refr_i;
                const float sin2_refr_i = 1.0f - cos2_refr_i;
                const float alpha2 = m_refraction_roughness * m_refraction_roughness;

                float factor = cos2_refr_i + sin2_refr_i / alpha2;
                factor *= factor;
                factor *= M_PI * alpha2;
                value += factor = 0.0f ? 0.0f : m_refraction_weight / factor;
            }
        }

        return value;
    }

    bool is_zero() const
    {
        return !(m_is_diffuse || m_is_translucent || m_is_reflective || m_is_refractive);
    }

    // TO-DO:
    // Roughness scaling after parameterization.

private:
    float m_diffuse_weight, m_reflection_weight, m_refraction_weight, m_translucency_weight;
    float m_reflection_roughness, m_refraction_roughness;
    float m_IOR;
    bool m_is_diffuse, m_is_translucent, m_is_reflective, m_is_refractive;

    Vector3f m_normal;
    Vector3f m_reflection_lobe;
    Vector3f m_refraction_lobe;
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_BSDFPROXY_H_ */