
#pragma once
#if !defined(__MITSUBA_RENDER_BSDFPROXY_H_)
#define __MITSUBA_RENDER_BSDFPROXY_H_

#include <mitsuba/mitsuba.h>

#include <mitsuba/productguiding/vcl-v2/vectorclass.h>

MTS_NAMESPACE_BEGIN

// SIMD Vector constants
Vec8f Zero_SIMD(0.0f);
Vec8f One_SIMD(1.0f);
Vec8f Two_SIMD(2.0f);
Vec8f One_Half_SIMD(0.5f);
Vec8f PI_SIMD(M_PI);
Vec8f Epsilon_SIMD(0.0001f);

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
        const float eta)
    {
        const float old_weight = m_refraction_weight;
        m_refraction_weight += refraction_weight;
        const float inv_weight = m_refraction_weight > 0.0f ? 1.0f / m_refraction_weight : 0.0f;
        m_refraction_roughness = old_weight * inv_weight * m_refraction_roughness + refraction_weight * inv_weight * roughness;

        m_eta = eta; // TO-DO: handle appropriately
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
        m_refraction_lobe = refract(outgoing, m_normal, m_eta);

        // Roughness correction.
        m_reflection_roughness *= 2.0f;
        const float cos_nt = std::abs(dot(m_normal, m_refraction_lobe));
        const float cos_no = std::abs(dot(m_normal, outgoing));
        const float scale_factor_refraction = (cos_nt + m_eta * cos_no) / cos_nt;
        m_refraction_roughness *= scale_factor_refraction;

        // Init SIMD
        m_normal_x = Vec8f(m_normal.x);
        m_normal_y = Vec8f(m_normal.y);
        m_normal_z = Vec8f(m_normal.z);
        m_diffuse_weight_SIMD = Vec8f(m_diffuse_weight);

        if (m_is_reflective)
        {
            m_reflection_lobe_x = Vec8f(m_reflection_lobe.x);
            m_reflection_lobe_y = Vec8f(m_reflection_lobe.y);
            m_reflection_lobe_z = Vec8f(m_reflection_lobe.z);
            m_reflection_weight_SIMD = Vec8f(m_reflection_weight);
            m_reflection_roughness_SIMD = Vec8f(m_reflection_roughness);
        }

        if (m_is_refractive)
        {
            m_refraction_lobe_x = Vec8f(m_refraction_lobe.x);
            m_refraction_lobe_y = Vec8f(m_refraction_lobe.y);
            m_refraction_lobe_z = Vec8f(m_refraction_lobe.z);
            m_refraction_weight_SIMD = Vec8f(m_refraction_weight);
            m_refraction_roughness_SIMD = Vec8f(m_refraction_roughness);
        }

        if (m_is_translucent)
        {
            m_translucency_weight_SIMD = Vec8f(m_translucency_weight);
        }

    }

    inline void get_lobes(Vector3f& diffuse_lobe, Vector3f& translucency_lobe, Vector3f& reflection_lobe, Vector3f& refraction_lobe) const
    {
        diffuse_lobe = m_normal;
        translucency_lobe = -m_normal;
        reflection_lobe = m_reflection_lobe;
        refraction_lobe = m_refraction_lobe;
    }

    inline Vec8f ggx_lobe_incoming_simd(const Vec8f& cos_lobe_in, const Vec8f& weight, const Vec8f& alpha) const
    {
        const Vec8f cos2_lobe_in = cos_lobe_in * cos_lobe_in;
        const Vec8f sin2_lobe_in = One_SIMD - cos2_lobe_in;
        const Vec8f alpha2 = alpha * alpha;

        Vec8f factor = cos2_lobe_in + sin2_lobe_in / alpha2;
        factor *= factor;
        factor *= PI_SIMD * alpha2;
        return select(factor == Zero_SIMD, Zero_SIMD, weight / factor);
    }

    inline float ggx_lobe_incoming(const float cos_lobe_in, const float weight, const float alpha) const
    {
        const float cos2_lobe_in = cos_lobe_in * cos_lobe_in;
        const float sin2_lobe_in = 1.0f - cos2_lobe_in;
        const float alpha2 = alpha * alpha;

        float factor = cos2_lobe_in + sin2_lobe_in / alpha2;
        factor *= factor;
        factor *= M_PI * alpha2;
        return factor == 0.0f ? 0.0f : weight / factor;
    }

    inline Vec8f dot_simd(const Vec8f &a_x, const Vec8f &a_y, const Vec8f &a_z,
                     const Vec8f &b_x, const Vec8f &b_y, const Vec8f &b_z) const
    {
        return a_x * b_x + a_y * b_y + a_z * b_z;
    }

    inline Vec8f evaluate_simd(
            const Vec8f &incoming_x, const Vec8f &incoming_y, const Vec8f &incoming_z, const Vec8f& diffuse_cosines, const Vec8f& reflection_cosines) const
    {
        Vec8f value(Zero_SIMD);
        const Vec8f cos_ni = min(dot_simd(m_normal_x, m_normal_y, m_normal_z, incoming_x, incoming_y, incoming_z) + diffuse_cosines, One_SIMD);
        // const Vec8f cos_ni = dot_simd(m_normal_x, m_normal_y, m_normal_z, incoming_x, incoming_y, incoming_z);
        const Vec8f cos_negni = -cos_ni;

        if (m_is_diffuse)
        {
            value += m_diffuse_weight_SIMD * max(cos_ni, Zero_SIMD);
        }
        if (m_is_translucent)
        {
            value += m_translucency_weight_SIMD * max(cos_negni, Zero_SIMD);
        }
        auto mask = cos_ni > Zero_SIMD;
        if (m_is_reflective && !horizontal_and(~mask))
        {
            const Vec8f cos_refl_i = min(dot_simd(m_reflection_lobe_x, m_reflection_lobe_y, m_reflection_lobe_z, incoming_x, incoming_y, incoming_z) + reflection_cosines, One_SIMD);
            // const Vec8f cos_refl_i = dot_simd(m_reflection_lobe_x, m_reflection_lobe_y, m_reflection_lobe_z, incoming_x, incoming_y, incoming_z);

            mask = mask & (cos_refl_i > Zero_SIMD);
            if (!horizontal_and(~mask))
            {
                value = if_add(mask, value, ggx_lobe_incoming_simd(cos_refl_i, m_reflection_weight_SIMD, m_reflection_roughness_SIMD));
            }
        }
        if (m_is_refractive)
        {
            const Vec8f cos_refr_i = dot_simd(m_refraction_lobe_x, m_refraction_lobe_y, m_refraction_lobe_z, incoming_x, incoming_y, incoming_z);

            auto mask = cos_refr_i > Zero_SIMD;
            mask = mask & ((dot_simd(m_refraction_lobe_x, m_refraction_lobe_y, m_refraction_lobe_z, m_normal_x, m_normal_y, m_normal_z)
                            * dot_simd(incoming_x, incoming_y, incoming_z, m_normal_x, m_normal_y, m_normal_z)) > Epsilon_SIMD);
            if (!horizontal_and(~mask))
            {
                value = if_add(mask, value, ggx_lobe_incoming_simd(cos_refr_i, m_refraction_weight_SIMD, m_refraction_roughness_SIMD));
            }
        }

        return value;
    }

    inline float evaluate(
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
                value += ggx_lobe_incoming(cos_refl_i, m_reflection_weight, m_reflection_roughness);
            }
        }
        if (m_is_refractive)
        {
            const float cos_refr_i = dot(m_refraction_lobe, incoming);

            if (cos_refr_i > 0.0f && dot(m_refraction_lobe, m_normal) * dot(incoming, m_normal) > 0.0001f)
            {
                value += ggx_lobe_incoming(cos_refr_i, m_refraction_weight, m_refraction_roughness);
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
    float m_eta;
    bool m_is_diffuse, m_is_translucent, m_is_reflective, m_is_refractive;

    Vector3f m_normal;
    Vector3f m_reflection_lobe;
    Vector3f m_refraction_lobe;

    Vec8f m_normal_x, m_normal_y, m_normal_z;
    Vec8f m_reflection_lobe_x, m_reflection_lobe_y, m_reflection_lobe_z;
    Vec8f m_refraction_lobe_x, m_refraction_lobe_y, m_refraction_lobe_z;
    Vec8f m_diffuse_weight_SIMD, m_reflection_weight_SIMD, m_refraction_weight_SIMD, m_translucency_weight_SIMD;
    Vec8f m_reflection_roughness_SIMD, m_refraction_roughness_SIMD;
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_BSDFPROXY_H_ */