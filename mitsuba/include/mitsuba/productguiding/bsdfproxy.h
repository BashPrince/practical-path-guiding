
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
Vec8f Small_Cosine(0.02f);
Vec8f Half_Cell(M_PI / 16.0f);

inline Vector canonicalToDir(Point2 p)
{
    const Float cosTheta = 2 * p.x - 1;
    const Float phi = 2 * M_PI * p.y;

    const Float sinTheta = sqrt(1 - cosTheta * cosTheta);
    Float sinPhi, cosPhi;
    math::sincos(phi, &sinPhi, &cosPhi);

    return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}

inline Point2 dirToCanonical(const Vector &d)
{
    if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z))
    {
        return {0, 0};
    }

    const Float cosTheta = std::min(std::max(d.z, -1.0f), 1.0f);
    Float phi = std::atan2(d.y, d.x);
    while (phi < 0)
        phi += 2.0 * M_PI;

    return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
}

class BSDFProxy
{
public:
    BSDFProxy()
        : m_diffuse_weight(0.0f), m_translucency_weight(0.0f), m_reflection_weight(0.0f), m_refraction_weight(0.0f), m_reflection_roughness(0.0f), m_refraction_roughness(0.0f)
    {
    }

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
        const Vector3f &shading_normal,
        const bool init_simd = false)
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

        if (!m_refraction_lobe.isZero())
            m_refraction_lobe = normalize(m_refraction_lobe);
        else
            m_is_refractive = false;

        // Roughness correction.
        m_reflection_roughness *= 2.0f;

        if (m_is_refractive)
        {
            const float cos_no = dot(m_normal, outgoing);
            const float cos_nt = dot(m_normal, m_refraction_lobe);
            const float abs_cos_no = std::abs(cos_no);
            const float abs_cos_nt = std::abs(cos_nt);
            const float eta = cos_no >= 0.0f ? m_eta : 1.0f / m_eta;
            const float scale_factor_refraction = abs_cos_nt != 0.0f ? (abs_cos_nt + eta * abs_cos_no) / (abs_cos_nt) : 1.0f;
            m_refraction_roughness *= scale_factor_refraction;
            m_refraction_roughness = std::max(std::min(m_refraction_roughness, 2.0f), 0.0f);
        }

        // Init SIMD
        if (init_simd)
        {
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
                m_cos_refl_n = dot_simd(m_reflection_lobe_x, m_reflection_lobe_y, m_reflection_lobe_z, m_normal_x, m_normal_y, m_normal_z);
            }

            if (m_is_refractive)
            {
                m_refraction_lobe_x = Vec8f(m_refraction_lobe.x);
                m_refraction_lobe_y = Vec8f(m_refraction_lobe.y);
                m_refraction_lobe_z = Vec8f(m_refraction_lobe.z);
                m_refraction_weight_SIMD = Vec8f(m_refraction_weight);
                m_refraction_roughness_SIMD = Vec8f(m_refraction_roughness);
                m_cos_refr_n = dot_simd(m_refraction_lobe_x, m_refraction_lobe_y, m_refraction_lobe_z, m_normal_x, m_normal_y, m_normal_z);
            }

            if (m_is_translucent)
            {
                m_translucency_weight_SIMD = Vec8f(m_translucency_weight);
            }
        }
    }

    inline void get_lobes(Vector3f &diffuse_lobe, Vector3f &translucency_lobe, Vector3f &reflection_lobe, Vector3f &refraction_lobe) const
    {
        diffuse_lobe = m_normal;
        translucency_lobe = -m_normal;
        reflection_lobe = m_reflection_lobe;
        refraction_lobe = m_refraction_lobe;
    }

    inline Vec8f ggx_lobe_incoming_simd(const Vec8f &cos_lobe_in, const Vec8f &weight, const Vec8f &alpha) const
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

    inline bool intersect_great_arc_segment(
        const Vector3f &vec,
        const Vector3f &lobe,
        const Vector3f &arc_from,
        const Vector3f &arc_to,
        Vector3f &intersection) const
    {
        const Vector3f p = cross(vec, lobe);
        const Vector3f q = cross(arc_from, arc_to);
        const Vector3f t = normalize(cross(p, q));

        const float s1 = dot(cross(vec, p), t);
        const float s2 = dot(cross(lobe, p), t);
        const float s3 = dot(cross(arc_from, q), t);
        const float s4 = dot(cross(arc_to, q), t);

        if (-s1 > 0 && s2 > 0 && -s3 > 0 && s4 > 0) {
            intersection = t;
            return true;
        }
        else if (-s1 < 0 && s2 < 0 && -s3 < 0 && s4 < 0) {
            intersection = -t;
            return true;
        }

        return false;
    }

    inline bool intersect_arc_latitude(
        const Vector3f &vec,
        const Vector3f &lobe,
        const Vector3f &arc_from,
        const Vector3f &arc_to,
        Vector3f &intersection) const
    {
        // Ray cone intersection
        Vector3f d = lobe - vec;
        const float tMax = d.length();
        d = normalize(d);

        float ox = vec.x, oy = vec.y, oz = vec.z;
        float dx = d.x, dy = d.y, dz = d.z;
        float height = arc_from.z;

        if (height < 0.0f)
        {
            oz = -oz;
            dz = -dz;
        }

        height = std::abs(height);
        oz += height;

        const float radius = Vector3f(arc_from.x, arc_from.y, 0.0f).length();
        float k = radius / height;
        k = k * k;
        float a = dx * dx + dy * dy - k * dz * dz;
        float b = 2 * (dx * ox + dy * oy - k * dz * (oz - height));
        float c = ox * ox + oy * oy - k * (oz - height) * (oz - height);

        // Solve quadratic equation for _t_ values
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1))
            return false;

        // Check quadric shape _t0_ and _t1_ for nearest intersection
        if (t0 > tMax || t1 <= 0)
            return false;

        float tShapeHit = t0;
        if (tShapeHit <= 0)
        {
            tShapeHit = t1;
            if (tShapeHit > tMax)
                return false;
        }

        intersection = normalize(vec + tShapeHit * d);

        return true;
    }

    inline Vector3f move_vec_towards_lobe(
        const Vector3f &vec,
        const Vector3f &lobe,
        const Vector3f &upper_left,
        const Vector3f &upper_right,
        const Vector3f &lower_left,
        const Vector3f &lower_right) const
    {
        bool found_intersection = false;
        Vector3f closest_intersection, next_intersection;

        bool found_next_intersection = intersect_great_arc_segment(vec, lobe, upper_left, lower_left, next_intersection);
        if (found_next_intersection && (!found_intersection || (next_intersection - vec).lengthSquared() < (closest_intersection - vec).lengthSquared())) {
            found_intersection = true;
            closest_intersection = next_intersection;
        }

        found_next_intersection = intersect_great_arc_segment(vec, lobe, upper_right, lower_right, next_intersection);
        if (found_next_intersection && (!found_intersection || (next_intersection - vec).lengthSquared() < (closest_intersection - vec).lengthSquared())) {
            found_intersection = true;
            closest_intersection = next_intersection;
        }

        if ((upper_left - upper_right).lengthSquared() > 0.001f) {
            if (std::abs(upper_left.z) < 0.01f) // equator = great circle
            {
                found_next_intersection = intersect_great_arc_segment(vec, lobe, upper_left, upper_right, next_intersection);
                if (found_next_intersection && (!found_intersection || (next_intersection - vec).lengthSquared() < (closest_intersection - vec).lengthSquared()))
                {
                    found_intersection = true;
                    closest_intersection = next_intersection;
                }
            }
            else
            {
                found_next_intersection = intersect_arc_latitude(vec, lobe, upper_left, upper_right, next_intersection);
                if (found_next_intersection && (!found_intersection || (next_intersection - vec).lengthSquared() < (closest_intersection - vec).lengthSquared()))
                {
                    found_intersection = true;
                    closest_intersection = next_intersection;
                }
            }
        }

        if ((lower_left - lower_right).lengthSquared() > 0.001f) {
            if (std::abs(lower_left.z) < 0.01f) // equator = great circle
            {
                found_next_intersection = intersect_great_arc_segment(vec, lobe, lower_left, lower_right, next_intersection);
                if (found_next_intersection && (!found_intersection || (next_intersection - vec).lengthSquared() < (closest_intersection - vec).lengthSquared()))
                {
                    found_intersection = true;
                    closest_intersection = next_intersection;
                }
            }
            else
            {
                found_next_intersection = intersect_arc_latitude(vec, lobe, lower_left, lower_right, next_intersection);
                if (found_next_intersection && (!found_intersection || (next_intersection - vec).lengthSquared() < (closest_intersection - vec).lengthSquared()))
                {
                    found_intersection = true;
                    closest_intersection = next_intersection;
                }
            }
        }

        return found_intersection ? closest_intersection : lobe;
    }

    inline void move_vec_towards_lobe(
        Vec8f &vec_x, Vec8f &vec_y, Vec8f &vec_z,
        const Vec8f &cell_upper_left_x, const Vec8f &cell_upper_left_y, const Vec8f &cell_upper_left_z,
        const Vec8f &cell_upper_right_x, const Vec8f &cell_upper_right_y, const Vec8f &cell_upper_right_z,
        const Vec8f &cell_lower_left_x, const Vec8f &cell_lower_left_y, const Vec8f &cell_lower_left_z,
        const Vec8f &cell_lower_right_x, const Vec8f &cell_lower_right_y, const Vec8f &cell_lower_right_z,
        const Vector3f &lobe) const
    {
        for (size_t i = 0; i < Vec8f::size(); ++i) {
            const Vector3f v(vec_x[i], vec_y[i], vec_z[i]);
            const Vector3f upper_left(cell_upper_left_x[i], cell_upper_left_y[i], cell_upper_left_z[i]);
            const Vector3f upper_right(cell_upper_right_x[i], cell_upper_right_y[i], cell_upper_right_z[i]);
            const Vector3f lower_left(cell_lower_left_x[i], cell_lower_left_y[i], cell_lower_left_z[i]);
            const Vector3f lower_right(cell_lower_right_x[i], cell_lower_right_y[i], cell_lower_right_z[i]);
            const Vector3f v_moved = move_vec_towards_lobe(v, lobe, upper_left, upper_right, lower_left, lower_right);

            // const Vector3f diff_lobe_v = lobe -v;
            // const float old_distance = diff_lobe_v.length();
            // const float new_distance = (lobe - v_moved).length();
            // const float distance_moved = (v - v_moved).length();

            // const Point2 old_cylindrical = dirToCanonical(v) * 16.0f;
            // const Point2 new_cylindrical = dirToCanonical(v_moved + 0.01f * (v - v_moved)) * 16.0f;

            // Point2i old_cell(old_cylindrical);
            // Point2i new_cell(new_cylindrical);

            // const Vector3f d = upper_left - lower_left;
            // const float l = d.length();

            // if (old_distance < new_distance || old_cell.x != new_cell.x || old_cell.y != new_cell.y)
            // {
            //     const Vector3f v_moved_local = move_vec_towards_lobe(v, lobe, upper_left, upper_right, lower_left, lower_right);
            // }

            vec_x.insert(i, v_moved.x);
            vec_y.insert(i, v_moved.y);
            vec_z.insert(i, v_moved.z);
        }
    }

    inline Vec8f evaluate_simd_accurate(
        const Vec8f &incoming_x, const Vec8f &incoming_y, const Vec8f &incoming_z,
        const Vec8f &cell_upper_left_x, const Vec8f &cell_upper_left_y, const Vec8f &cell_upper_left_z,
        const Vec8f &cell_upper_right_x, const Vec8f &cell_upper_right_y, const Vec8f &cell_upper_right_z,
        const Vec8f &cell_lower_left_x, const Vec8f &cell_lower_left_y, const Vec8f &cell_lower_left_z,
        const Vec8f &cell_lower_right_x, const Vec8f &cell_lower_right_y, const Vec8f &cell_lower_right_z) const
    {
        Vec8f value(Zero_SIMD);

        if (m_is_diffuse)
        {
            Vec8f in_towards_n_x = incoming_x;
            Vec8f in_towards_n_y = incoming_y;
            Vec8f in_towards_n_z = incoming_z;
            move_vec_towards_lobe(
                in_towards_n_x, in_towards_n_y, in_towards_n_z,
                cell_upper_left_x, cell_upper_left_y, cell_upper_left_z,
                cell_upper_right_x, cell_upper_right_y, cell_upper_right_z,
                cell_lower_left_x, cell_lower_left_y, cell_lower_left_z,
                cell_lower_right_x, cell_lower_right_y, cell_lower_right_z,
                m_normal);
            const Vec8f cos_ni = dot_simd(m_normal_x, m_normal_y, m_normal_z, in_towards_n_x, in_towards_n_y, in_towards_n_z);
            value += m_diffuse_weight_SIMD * select(cos_ni > Zero_SIMD, cos_ni, Zero_SIMD);
        }
        if (m_is_reflective)
        {
            Vec8f in_towards_refl_x = incoming_x;
            Vec8f in_towards_refl_y = incoming_y;
            Vec8f in_towards_refl_z = incoming_z;
            move_vec_towards_lobe(
                in_towards_refl_x, in_towards_refl_y, in_towards_refl_z,
                cell_upper_left_x, cell_upper_left_y, cell_upper_left_z,
                cell_upper_right_x, cell_upper_right_y, cell_upper_right_z,
                cell_lower_left_x, cell_lower_left_y, cell_lower_left_z,
                cell_lower_right_x, cell_lower_right_y, cell_lower_right_z,
                m_reflection_lobe);
            const Vec8f cos_refl_i = dot_simd(m_reflection_lobe_x, m_reflection_lobe_y, m_reflection_lobe_z, in_towards_refl_x, in_towards_refl_y, in_towards_refl_z);

            value += select(dot_simd(in_towards_refl_x, in_towards_refl_y, in_towards_refl_z, m_normal_x, m_normal_y, m_normal_z) * m_cos_refl_n > Zero_SIMD,
                            ggx_lobe_incoming_simd(
                                select(cos_refl_i > Zero_SIMD, cos_refl_i, Small_Cosine),
                                m_reflection_weight_SIMD,
                                m_reflection_roughness_SIMD),
                            Zero_SIMD);
        }
        if (m_is_refractive)
        {
            Vec8f in_towards_refr_x = incoming_x;
            Vec8f in_towards_refr_y = incoming_y;
            Vec8f in_towards_refr_z = incoming_z;
            move_vec_towards_lobe(
                in_towards_refr_x, in_towards_refr_y, in_towards_refr_z,
                cell_upper_left_x, cell_upper_left_y, cell_upper_left_z,
                cell_upper_right_x, cell_upper_right_y, cell_upper_right_z,
                cell_lower_left_x, cell_lower_left_y, cell_lower_left_z,
                cell_lower_right_x, cell_lower_right_y, cell_lower_right_z,
                m_refraction_lobe);
            const Vec8f cos_refr_i = dot_simd(m_reflection_lobe_x, m_reflection_lobe_y, m_reflection_lobe_z, in_towards_refr_x, in_towards_refr_y, in_towards_refr_z);

            value += select(dot_simd(in_towards_refr_x, in_towards_refr_y, in_towards_refr_z, m_normal_x, m_normal_y, m_normal_z) * m_cos_refr_n > Zero_SIMD,
                            ggx_lobe_incoming_simd(
                                select(cos_refr_i > Zero_SIMD, cos_refr_i, Small_Cosine),
                                m_refraction_weight_SIMD,
                                m_refraction_roughness_SIMD),
                            Zero_SIMD);
        }

        return value;
    }

    inline Vec8f evaluate_simd(
        const Vec8f &incoming_x, const Vec8f &incoming_y, const Vec8f &incoming_z) const
    {
        Vec8f value(Zero_SIMD);
        const Vec8f cos_ni = dot_simd(m_normal_x, m_normal_y, m_normal_z, incoming_x, incoming_y, incoming_z);

        if (m_is_diffuse)
        {
            value += m_diffuse_weight_SIMD * select(cos_ni > -Half_Cell, max(cos_ni, Small_Cosine), Zero_SIMD);
        }
        if (m_is_translucent)
        {
            value += m_translucency_weight_SIMD * select(cos_ni < Half_Cell, max(abs(cos_ni), Small_Cosine), Zero_SIMD);
        }
        if (m_is_reflective)
        {
            const Vec8f cos_refl_i = dot_simd(m_reflection_lobe_x, m_reflection_lobe_y, m_reflection_lobe_z, incoming_x, incoming_y, incoming_z);

            value += select((cos_ni * m_cos_refl_n > Zero_SIMD || abs(cos_ni) < Half_Cell) && cos_refl_i > -Half_Cell,
                            ggx_lobe_incoming_simd(
                                max(cos_refl_i, Small_Cosine),
                                m_reflection_weight_SIMD,
                                m_reflection_roughness_SIMD),
                            Zero_SIMD);
        }
        if (m_is_refractive)
        {
            const Vec8f cos_refr_i = dot_simd(m_refraction_lobe_x, m_refraction_lobe_y, m_refraction_lobe_z, incoming_x, incoming_y, incoming_z);

            value += select((cos_ni * m_cos_refr_n > Zero_SIMD || abs(cos_ni) < Half_Cell) && cos_refr_i > -Half_Cell,
                            ggx_lobe_incoming_simd(
                                max(cos_refr_i, Small_Cosine),
                                m_refraction_weight_SIMD,
                                m_refraction_roughness_SIMD),
                            Zero_SIMD);
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
        if (m_is_reflective)
        {
            const float cos_refl_i = dot(m_reflection_lobe, incoming);

            if (cos_refl_i > 0.0f && dot(m_reflection_lobe, m_normal) * dot(incoming, m_normal) > 0.0001f)
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

    float integrate(const Point2 &from, const Point2 &to) const
    {

        const size_t steps = 3;
        const Vector2 diff = to - from;
        const float step_size_x = diff.x / steps;
        const float step_size_y = diff.y / steps;

        const Point2 start = from + Point2(0.5f * step_size_x, 0.5f * step_size_y);

        float sum = 0.0f;

        for (size_t y = 0; y < steps; ++y)
        {
            for (size_t x = 0; x < steps; ++x)
            {
                const Point2 p = start + Point2(x * step_size_x, y * step_size_y);
                const Vector3f dir = canonicalToDir(p);
                sum += evaluate(dir);
            }
        }

        return sum / (steps * steps);
    }

    // TO-DO:
    // Roughness scaling after parameterization.

    // private:
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

    Vec8f m_cos_refl_n, m_cos_refr_n;
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_BSDFPROXY_H_ */