/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob
    Copyright (c) 2017 by ETH Zurich, Thomas Mueller.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/productguiding/bsdfproxy.h>
#include <mitsuba/core/warp.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>

#include <immintrin.h>
#include <mitsuba/productguiding/vcl-v2/vectorclass.h>
#include <mitsuba/productguiding/vcl-v2/vectormath_trig.h>

#include <mutex>

#define BUILD_SIMD_PRODUCT
#define PROXYWIDTH 16
#define LOAD_INCOMING_ARRAY

#ifdef LOAD_INCOMING_ARRAY
#if PROXYWIDTH == 16
    #include <mitsuba/productguiding/proxyarrays_16.h>
#elif PROXYWIDTH == 8
    #include <mitsuba/productguiding/proxyarrays_8.h>
#else
    #error(ProxyWidth neither 8 nor 16)
#endif
#endif

MTS_NAMESPACE_BEGIN

class BlobWriter {
public:
    BlobWriter(const std::string& filename)
        : f(filename, std::ios::out | std::ios::binary) {
    }

    template <typename Type>
    typename std::enable_if<std::is_standard_layout<Type>::value, BlobWriter&>::type
        operator << (Type Element) {
        Write(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void Write(T* Src, size_t Size) {
        f.write(reinterpret_cast<const char*>(Src), Size * sizeof(T));
    }

private:
    std::ofstream f;
};

static void addToAtomicFloat(std::atomic<Float>& var, Float val) {
    auto current = var.load();
    while (!var.compare_exchange_weak(current, current + val));
}

inline Float logistic(Float x) {
    return 1 / (1 + std::exp(-x));
}

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

inline void canonicalToDir_simd(const Vec8f &p_x, const Vec8f &p_y, Vec8f &outDir_x, Vec8f &outDir_y, Vec8f &outDir_z)
{
    const Vec8f cosTheta = Two_SIMD * p_x - One_SIMD;
    const Vec8f phi = Two_SIMD * Vec8f(M_PI) * p_y;

    const Vec8f sinTheta = sqrt(One_SIMD - cosTheta * cosTheta);
    Vec8f cosPhi;
    const Vec8f sinPhi = sincos(&cosPhi, phi);

    outDir_x = sinTheta * cosPhi;
    outDir_y = sinTheta * sinPhi;
    outDir_z = cosTheta;
}

void make_incoming_arrays(const size_t ProxyWidth)
{
    std::vector<Vector3f> incoming_dirs(ProxyWidth * ProxyWidth);

    const float inv_width = 1.0f / ProxyWidth;
    for (size_t y = 0; y < ProxyWidth; ++y)
    {
        for (size_t x = 0; x < ProxyWidth; ++x)
        {
            const Point2u pixel(x, y);
            const Point2f cylindrical_direction(
                (x + 0.5f) * inv_width,
                (y + 0.5f) * inv_width);

            const Vector3f incoming = canonicalToDir(cylindrical_direction);
            const size_t index = pixel.y * ProxyWidth + pixel.x;
            incoming_dirs[index] = incoming;
        }
    }

    // Save AoS arrays
    std::string filename_base = "/mnt/Data/_Programming/PracticalPathGuiding/WorldDir_";
    filename_base += std::to_string(ProxyWidth) + "_";

    std::ofstream file_x(filename_base + "x");
    std::ofstream file_y(filename_base + "y");
    std::ofstream file_z(filename_base + "z");
    file_x << std::setprecision(8) << "{\n";
    file_y << std::setprecision(8) << "{\n";
    file_z << std::setprecision(8) << "{\n";

    for (size_t y = 0; y < ProxyWidth; ++y)
    {
        for (size_t x = 0; x < ProxyWidth; ++x)
        {
            const Point2u pixel(x, y);
            const size_t index = pixel.y * ProxyWidth + pixel.x;
            file_x << incoming_dirs[index].x << ", ";
            file_y << incoming_dirs[index].y << ", ";
            file_z << incoming_dirs[index].z << ", ";
        }
        file_x << "\n";
        file_y << "\n";
        file_z << "\n";
    }
    file_x << "}";
    file_y << "}";
    file_z << "}";
    file_x.close();
    file_y.close();
    file_z.close();

    // Save SoA array
    std::ofstream file_soa(filename_base + "SoA");
    file_soa << std::setprecision(8) << "{\n";

    for (size_t y = 0; y < ProxyWidth; ++y)
    {
        for (size_t x = 0; x < ProxyWidth; x += 8)
        {
            const Point2u pixel(x, y);
            const size_t index = pixel.y * ProxyWidth + pixel.x;

            for (size_t i = 0; i < 8; ++i)
                file_soa << incoming_dirs[index + i].x << ", ";
            
            file_soa << "\n";
            
            for (size_t i = 0; i < 8; ++i)
                file_soa << incoming_dirs[index + i].y << ", ";
            
            file_soa << "\n";

            for (size_t i = 0; i < 8; ++i)
                file_soa << incoming_dirs[index + i].z << ", ";
            
            file_soa << "\n";
        }
    }
    file_soa << "}";
    file_soa.close();

    // Save scalar array
    std::ofstream file_scalar(filename_base + "scalar");
    file_scalar << std::setprecision(8) << "{\n";

    for (size_t y = 0; y < ProxyWidth; ++y)
    {
        for (size_t x = 0; x < ProxyWidth; ++x)
        {
            const Point2u pixel(x, y);
            const size_t index = pixel.y * ProxyWidth + pixel.x;
            file_scalar << incoming_dirs[index].x << ", ";
            file_scalar << incoming_dirs[index].y << ", ";
            file_scalar << incoming_dirs[index].z << ", ";
            file_scalar << "\n";
        }
    }
    file_scalar << "}";
    file_scalar.close();
}

alignas(32) float cosines[256 * 256];
bool cosines_loaded = false;
std::mutex cosine_mutex;

// Implements the stochastic-gradient-based Adam optimizer [Kingma and Ba 2014]
class AdamOptimizer {
public:
    AdamOptimizer(Float learningRate, Float learningRateProduct, int batchSize = 1, Float epsilon = 1e-08f, Float beta1 = 0.9f, Float beta2 = 0.999f)
    {
        m_hparams = {learningRate, learningRateProduct, batchSize, epsilon, beta1, beta2};
	}

    AdamOptimizer& operator=(const AdamOptimizer& arg) {
        m_state = arg.m_state;
        m_product_state = arg.m_product_state;
        m_hparams = arg.m_hparams;
        return *this;
    }

    AdamOptimizer(const AdamOptimizer& arg) {
        *this = arg;
    }

    void append(Float gradient, Float statisticalWeight) {
        m_state.batchGradient += gradient * statisticalWeight;
        m_state.batchAccumulation += statisticalWeight;

        if (m_state.batchAccumulation > m_hparams.batchSize) {
            step(m_state.batchGradient / m_state.batchAccumulation);

            m_state.batchGradient = 0;
            m_state.batchAccumulation = 0;
        }
    }

    void append_product(Vector2 gradient, Float statisticalWeight) {
        m_product_state.batchGradient += gradient * statisticalWeight;
        m_product_state.batchAccumulation += statisticalWeight;

        if (m_product_state.batchAccumulation > m_hparams.batchSize) {
            step_product(m_product_state.batchGradient / m_product_state.batchAccumulation);

            m_product_state.batchGradient = Vector2(0.0f);
            m_product_state.batchAccumulation = 0;
        }
    }

    void step(Float gradient) {
        ++m_state.iter;

        Float actualLearningRate = m_hparams.learningRate * std::sqrt(1 - std::pow(m_hparams.beta2, m_state.iter)) / (1 - std::pow(m_hparams.beta1, m_state.iter));
        m_state.firstMoment = m_hparams.beta1 * m_state.firstMoment + (1 - m_hparams.beta1) * gradient;
        m_state.secondMoment = m_hparams.beta2 * m_state.secondMoment + (1 - m_hparams.beta2) * gradient * gradient;
        m_state.variable -= actualLearningRate * m_state.firstMoment / (std::sqrt(m_state.secondMoment) + m_hparams.epsilon);

        // Clamp the variable to the range [-20, 20] as a safeguard to avoid numerical instability:
        // since the sigmoid involves the exponential of the variable, value of -20 or 20 already yield
        // in *extremely* small and large results that are pretty much never necessary in practice.
        m_state.variable = std::min(std::max(m_state.variable, -20.0f), 20.0f);
    }

    void step_product(Vector2 gradient) {
        ++m_product_state.iter;

        Float actualLearningRate = m_hparams.learningRateProduct * std::sqrt(1 - std::pow(m_hparams.beta2, m_product_state.iter)) / (1 - std::pow(m_hparams.beta1, m_product_state.iter));
        m_product_state.firstMoment = m_hparams.beta1 * m_product_state.firstMoment + (1 - m_hparams.beta1) * gradient;
        m_product_state.secondMoment = m_hparams.beta2 * m_product_state.secondMoment + (1 - m_hparams.beta2) * Vector2(gradient.x * gradient.x, gradient.y + gradient.y);
        m_product_state.variable.x -= actualLearningRate * m_product_state.firstMoment.x / (std::sqrt(m_product_state.secondMoment.x) + m_hparams.epsilon);
        m_product_state.variable.y -= actualLearningRate * m_product_state.firstMoment.y / (std::sqrt(m_product_state.secondMoment.y) + m_hparams.epsilon);

        // Clamp the variable to the range [-20, 20] as a safeguard to avoid numerical instability:
        // since the sigmoid involves the exponential of the variable, value of -20 or 20 already yield
        // in *extremely* small and large results that are pretty much never necessary in practice.
        m_product_state.variable = Vector2(
            math::clamp(m_product_state.variable.x, -20.0f, 20.0f),
            math::clamp(m_product_state.variable.y, -20.0f, 20.0f));
    }

    Float variable() const {
        return m_state.variable;
    }

    Vector2 variable_combined() const {
        return m_product_state.variable;
    }

private:
    struct State {
        int iter = 0;
        Float firstMoment = 0;
        Float secondMoment = 0;
        Float variable = 0;

        Float batchAccumulation = 0;
        Float batchGradient = 0;
    } m_state;

    struct ProductState {
        int iter = 0;
        Vector2 firstMoment = Vector2(0.0f);
        Vector2 secondMoment = Vector2(0.0f);
        Vector2 variable = Vector2(-0.693147181f, 0.0f); // Results in equal probability for all sampling methods

        Float batchAccumulation = 0;
        Vector2 batchGradient = Vector2(0.0f);
    } m_product_state;

    struct Hyperparameters {
        Float learningRate;
        Float learningRateProduct;
        int batchSize;
        Float epsilon;
        Float beta1;
        Float beta2;
    } m_hparams;
};

enum class ESampleCombination {
    EDiscard,
    EDiscardWithAutomaticBudget,
    EInverseVariance,
};

enum class EBsdfSamplingFractionLoss {
    ENone,
    EKL,
    EVariance,
};

enum class EGuidingMode {
    ERadiance,
    EProduct,
    ECombined
};

enum class ESpatialFilter {
    ENearest,
    EStochasticBox,
    EBox,
};

enum class EDirectionalFilter {
    ENearest,
    EBox,
};

class QuadTreeNode;

template <typename Predicate>
int FindInterval(int size, const Predicate &pred)
{
    int first = 0, len = size;
    while (len > 0)
    {
        int half = len >> 1, middle = first + half;
        // Bisect range based on value of _pred_ at _middle_
        if (pred(middle))
        {
            first = middle + 1;
            len -= half + 1;
        }
        else
            len = half;
    }
    return math::clamp(first - 1, 0, size - 2);
}

class RadianceProxy
{
public:
    RadianceProxy();

    RadianceProxy(
        const RadianceProxy &other);

    void build(
        const std::vector<QuadTreeNode> *quadtree_nodes,
        const float radiance_scale);

    void build_product(
        BSDFProxy &bsdf_proxy,
        const Vector3f &outgoing,
        const Vector3f &shading_normal);

    void build_product_scalar(
        BSDFProxy &bsdf_proxy,
        const Vector3f &outgoing,
        const Vector3f &shading_normal);

    void build_product_simd(
        BSDFProxy &bsdf_proxy,
        const Vector3f &outgoing,
        const Vector3f &shading_normal);
    
    void clear();

    float proxy_radiance(
        const Vector3f &direction) const;

    float sample(
        Sampler *sampler,
        Vector3f &direction) const;

    float pdf(
        const Vector3f &direction) const;

    bool is_built() const;

    void set_maps(const RadianceProxy& other);

    static const size_t ProxyWidth = PROXYWIDTH;

    template <size_t Width>
    class ImageImportanceSampler
    {
    public:
        void build()
        {
            // SIMD version
#ifdef BUILD_SIMD_PRODUCT
            float marginal_sum = 0.0f;
            size_t start_index = (Width - 1) * Width;

            for (size_t i = 0; i < Width; ++i)
            {
                marginal_sum += distribution[start_index + i];
                marginalDistribution[i] = marginal_sum;
            }

            totalRadiance = marginalDistribution[Width - 1];
            if (totalRadiance <= 0.0f)
                isZero = true;
#else

            // Scalar version
            float marginal_sum = 0.0f;
            size_t start_index = Width - 1;

            for (size_t i = 0; i < Width; ++i)
            {
                marginal_sum += distribution[start_index + i * Width];
                marginalDistribution[i] = marginal_sum;
            }

            totalRadiance = marginalDistribution[Width - 1];
            if (totalRadiance <= 0.0f)
                isZero = true;
#endif
        }

        void sample(const Point2 &s, Point2u &pixel) const
        {
            if (isZero)
            {
                pixel.x = (unsigned int)(s.x * Width);
                pixel.y = (unsigned int)(s.y * Width);
            }
#ifdef BUILD_SIMD_PRODUCT
            const float s_x_scaled = s.x * totalRadiance;
            size_t x = 0;
            while (s_x_scaled > marginalDistribution[x])
                ++x;

            assert(x < Width);

            size_t y = 0;
            const float s_y_scaled = s.y * distribution[(Width - 1) * Width + x];
            while (s_y_scaled > distribution[y * Width + x])
                ++y;
#else
            const float s_y_scaled = s.y * totalRadiance;
            size_t y = 0;
            while (s_y_scaled > marginalDistribution[y])
                ++y;

            assert(y < Width);

            size_t x = 0;
            const size_t row_index = y * Width;
            const float s_x_scaled = s.x * distribution[row_index + Width - 1];
            while (s_x_scaled > distribution[row_index + x])
                ++x;
#endif
            
            pixel.x = x;
            pixel.y = y;
        }

        float* get_distribution()
        {
            return distribution.data();
        }

        inline float get_total_radiance() const
        {
            return totalRadiance;
        }

        inline bool is_zero() const
        {
            return isZero;
        }

    private:
        std::array<float, PROXYWIDTH> marginalDistribution;
        std::array<float, PROXYWIDTH * PROXYWIDTH> distribution;
        float totalRadiance;
        bool isZero = false;
    };

    ImageImportanceSampler<PROXYWIDTH> m_image_importance_sampler;

    std::array<float, PROXYWIDTH * PROXYWIDTH> m_map;
    const std::array<float, PROXYWIDTH * PROXYWIDTH>* m_parent_map;

    std::shared_ptr<
        std::array<const QuadTreeNode *, PROXYWIDTH * PROXYWIDTH>>
        m_quadtree_strata;

    bool m_product_is_built;
    bool m_is_built;
    const std::vector<QuadTreeNode> *m_quadtree_nodes;
};

class QuadTreeNode {
public:
    QuadTreeNode() {
        m_children = {};
        for (size_t i = 0; i < m_sum.size(); ++i) {
            m_sum[i].store(0, std::memory_order_relaxed);
        }
    }

    void setSum(int index, Float val) {
        m_sum[index].store(val, std::memory_order_relaxed);
    }

    Float sum(int index) const {
        return m_sum[index].load(std::memory_order_relaxed);
    }

    void copyFrom(const QuadTreeNode& arg) {
        for (int i = 0; i < 4; ++i) {
            setSum(i, arg.sum(i));
            m_children[i] = arg.m_children[i];
        }
    }

    QuadTreeNode(const QuadTreeNode& arg) {
        copyFrom(arg);
    }

    QuadTreeNode& operator=(const QuadTreeNode& arg) {
        copyFrom(arg);
        return *this;
    }

    void setChild(int idx, uint16_t val) {
        m_children[idx] = val;
    }

    uint16_t child(int idx) const {
        return m_children[idx];
    }

    void setSum(Float val) {
        for (int i = 0; i < 4; ++i) {
            setSum(i, val);
        }
    }

    int childIndex(Point2& p) const {
        int res = 0;
        for (int i = 0; i < Point2::dim; ++i) {
            if (p[i] < 0.5f) {
                p[i] *= 2;
            } else {
                p[i] = (p[i] - 0.5f) * 2;
                res |= 1 << i;
            }
        }

        return res;
    }

    // Evaluates the directional irradiance *sum density* (i.e. sum / area) at a given location p.
    // To obtain radiance, the sum density (result of this function) must be divided
    // by the total statistical weight of the estimates that were summed up.
    Float eval(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 4 * sum(index);
        } else {
            return 4 * nodes[child(index)].eval(p, nodes);
        }
    }

    Float pdf(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (!(sum(index) > 0)) {
            return 0;
        }

        const Float factor = 4 * sum(index) / (sum(0) + sum(1) + sum(2) + sum(3));
        if (isLeaf(index)) {
            return factor;
        } else {
            return factor * nodes[child(index)].pdf(p, nodes);
        }
    }

    int depthAt(Point2& p, const std::vector<QuadTreeNode>& nodes) const {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 1;
        } else {
            return 1 + nodes[child(index)].depthAt(p, nodes);
        }
    }

    Point2 sample(Sampler* sampler, const std::vector<QuadTreeNode>& nodes) const {
        int index = 0;

        Float topLeft = sum(0);
        Float topRight = sum(1);
        Float partial = topLeft + sum(2);
        Float total = partial + topRight + sum(3);

        // Should only happen when there are numerical instabilities.
        if (!(total > 0.0f)) {
            return sampler->next2D();
        }

        Float boundary = partial / total;
        Point2 origin = Point2{0.0f, 0.0f};

        Float sample = sampler->next1D();

        if (sample < boundary) {
            SAssert(partial > 0);
            sample /= boundary;
            boundary = topLeft / partial;
        } else {
            partial = total - partial;
            SAssert(partial > 0);
            origin.x = 0.5f;
            sample = (sample - boundary) / (1.0f - boundary);
            boundary = topRight / partial;
            index |= 1 << 0;
        }

        if (sample < boundary) {
            sample /= boundary;
        } else {
            origin.y = 0.5f;
            sample = (sample - boundary) / (1.0f - boundary);
            index |= 1 << 1;
        }

        if (isLeaf(index)) {
            return origin + 0.5f * sampler->next2D();
        } else {
            return origin + 0.5f * nodes[child(index)].sample(sampler, nodes);
        }
    }

    void record(Point2& p, Float irradiance, std::vector<QuadTreeNode>& nodes) {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        int index = childIndex(p);

        if (isLeaf(index)) {
            addToAtomicFloat(m_sum[index], irradiance);
        } else {
            nodes[child(index)].record(p, irradiance, nodes);
        }
    }

    Float computeOverlappingArea(const Point2& min1, const Point2& max1, const Point2& min2, const Point2& max2) {
        Float lengths[2];
        for (int i = 0; i < 2; ++i) {
            lengths[i] = std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1];
    }

    void record(const Point2& origin, Float size, Point2 nodeOrigin, Float nodeSize, Float value, std::vector<QuadTreeNode>& nodes) {
        Float childSize = nodeSize / 2;
        for (int i = 0; i < 4; ++i) {
            Point2 childOrigin = nodeOrigin;
            if (i & 1) { childOrigin[0] += childSize; }
            if (i & 2) { childOrigin[1] += childSize; }

            Float w = computeOverlappingArea(origin, origin + Point2(size), childOrigin, childOrigin + Point2(childSize));
            if (w > 0.0f) {
                if (isLeaf(i)) {
                    addToAtomicFloat(m_sum[i], value * w);
                } else {
                    nodes[child(i)].record(origin, size, childOrigin, childSize, value, nodes);
                }
            }
        }
    }

    bool isLeaf(int index) const {
        return child(index) == 0;
    }

    // Ensure that each quadtree node's sum of irradiance estimates
    // equals that of all its children.
    void build(std::vector<QuadTreeNode>& nodes) {
        for (int i = 0; i < 4; ++i) {
            // During sampling, all irradiance estimates are accumulated in
            // the leaves, so the leaves are built by definition.
            if (isLeaf(i)) {
                continue;
            }

            QuadTreeNode& c = nodes[child(i)];

            // Recursively build each child such that their sum becomes valid...
            c.build(nodes);

            // ...then sum up the children's sums.
            Float sum = 0;
            for (int j = 0; j < 4; ++j) {
                sum += c.sum(j);
            }
            setSum(i, sum);
        }
    }

    void build_radiance_proxy(
        RadianceProxy & radiance_proxy,
        const float radiance_factor,
        const size_t end_level,
        const std::vector<QuadTreeNode>* nodes,
        const Point2u origin = Point2u(0, 0),
        const size_t depth = 1) const
    {
        for (int i = 0; i < 4; ++i)
        {
            const Point2u sub_node_offset(i % 2, i >> 1);
            if (depth == end_level || isLeaf(i))
            {
                // Write node i to map.
                const size_t level_diff = end_level - depth;
                size_t width = 1;
                Point2u pixel_origin = (unsigned int)2 * origin + sub_node_offset;

                for (size_t i = 0; i < level_diff; ++i)
                {
                    width *= 2;
                    pixel_origin *= 2;
                }

                const float radiance = 4.0f * radiance_factor * sum(i);

                for (size_t y = 0; y < width; ++y)
                {
                    for (size_t x = 0; x < width; ++x)
                    {
                        const Point2u pixel = pixel_origin + Point2u(x, y);
                        const size_t pixel_index = pixel.y * RadianceProxy::ProxyWidth + pixel.x;

                        assert(pixel_index >= 0);
                        assert(pixel_index < RadianceProxy::ProxyWidth * RadianceProxy::ProxyWidth);
                        radiance_proxy.m_map[pixel_index] = radiance;

                        assert(radiance_proxy.m_quadtree_strata != nullptr);
                        (*radiance_proxy.m_quadtree_strata)[pixel_index] = !isLeaf(i) ? &((*nodes)[child(i)]) : nullptr;
                    }
                }
            }
            else
            {
                // Recursively write node i's children to map.
                const Point2u sub_node_origin = (unsigned int)2 * origin + sub_node_offset;
                (*nodes)[child(i)].build_radiance_proxy(
                    radiance_proxy,
                    radiance_factor * 4.0f,
                    end_level,
                    nodes,
                    sub_node_origin,
                    depth + 1);
            }
        }
    }

    float radiance(Point2& dir, const std::vector<QuadTreeNode>& nodes) const
    {
        const int index = childIndex(dir);
        if (isLeaf(index))
        {
            return 4 * sum(index);
        }
        else
        {
            return 4 * nodes[child(index)].radiance(dir, nodes);
        }
    }

private:
    std::array<std::atomic<Float>, 4> m_sum;
    std::array<uint16_t, 4> m_children;
};

RadianceProxy::RadianceProxy()
    : m_product_is_built(false)
    , m_is_built(false)
    , m_quadtree_nodes(nullptr)
{}

RadianceProxy::RadianceProxy(
    const RadianceProxy&                    other)
    : m_map(other.m_map)
    , m_quadtree_strata(other.m_quadtree_strata)
    , m_product_is_built(false)
    , m_is_built(other.m_is_built)
    , m_quadtree_nodes(other.m_quadtree_nodes)
{}

void RadianceProxy::build(
    const std::vector<QuadTreeNode>* quadtree_nodes,
    const float radiance_scale)
{
    m_quadtree_nodes = quadtree_nodes;
    m_quadtree_strata = std::make_shared<std::array<const QuadTreeNode *, PROXYWIDTH * PROXYWIDTH>>();

    size_t end_level = 0;
    size_t map_width = ProxyWidth;

    while (map_width > 1)
    {
        ++end_level;
        map_width = map_width >> 1;
    }

    (*m_quadtree_nodes)[0].build_radiance_proxy(
        (*this),
        radiance_scale,
        end_level,
        m_quadtree_nodes);

    for (float &pixel_val : m_map)
    {
        if (pixel_val < 0.0f || std::isnan(pixel_val) || std::isinf(pixel_val))
            pixel_val = 0.0f;
    }

    m_is_built = true;
}

// Return cos theta of v to boundary in direction dest
float cos_theta_to_boundary(const Point2u &p, const Point2u &dest, const float ProxyWidth)
{
    if (p == dest)
        return 1.0f;

    const float inv_width = 1.0f / ProxyWidth;
    const Point2f cylindrical_direction_p(
        (p.x + 0.5f) * inv_width,
        (p.y + 0.5f) * inv_width);

    const Point2f cylindrical_direction_dest(
        (dest.x + 0.5f) * inv_width,
        (dest.y + 0.5f) * inv_width);

    const Vector3f incoming = canonicalToDir(cylindrical_direction_p);
    const Vector3f destination = canonicalToDir(cylindrical_direction_dest);

    // For opposite sides return 0
    if ((incoming + destination).length() < 0.001)
        return 0.0f;

    const Vector3f diff = destination - incoming;

    Vector3f current = incoming;
    Point2u currentPixel = p;
    Vector3f step = diff * 0.01f;

    while (currentPixel == p)
    {
        // Step along line towards dest
        current += step;

        // Reproject onto sphere
        const Vector3f s = normalize(current);
        const Point2f s_canonical = dirToCanonical(s) * static_cast<float>(ProxyWidth);

        // Pixel coords
        currentPixel = Point2u(s_canonical.x, s_canonical.y);

        currentPixel.x = std::min(currentPixel.x, static_cast<unsigned int>(ProxyWidth - 1));
        currentPixel.y = std::min(currentPixel.y, static_cast<unsigned int>(ProxyWidth - 1));
    }

    return dot(destination, normalize(current)) - dot(destination, incoming);
}

void RadianceProxy::build_product(
    BSDFProxy &bsdf_proxy,
    const Vector3f &outgoing,
    const Vector3f &shading_normal)
{
#ifdef BUILD_SIMD_PRODUCT
    build_product_simd(bsdf_proxy, outgoing, shading_normal);
#else
    build_product_scalar(bsdf_proxy, outgoing, shading_normal);
#endif
}

void RadianceProxy::build_product_scalar(
    BSDFProxy& bsdf_proxy,
    const Vector3f& outgoing,
    const Vector3f& shading_normal)
{
    assert(m_is_built);

    if (m_product_is_built)
        return;

    bsdf_proxy.finish_parameterization(outgoing, shading_normal);
    m_product_is_built = true;

    float *distribution = m_image_importance_sampler.get_distribution();

    const float inv_width = 1.0f / ProxyWidth;
    for (size_t y = 0; y < ProxyWidth; ++y)
    {
        float distribution_sum = 0.0f;
        for (size_t x = 0; x < ProxyWidth; ++x)
        {
            const Point2u pixel(x, y);
            const size_t index = pixel.y * ProxyWidth + pixel.x;
            
#ifdef LOAD_INCOMING_ARRAY
            const Vector3f incoming(worldDir_scalar[3 * index], worldDir_scalar[3 * index + 1], worldDir_scalar[3 * index + 2]);
#else
            const Point2f cylindrical_direction(
                (x + 0.5f) * inv_width,
                (y + 0.5f) * inv_width);
            const Vector3f incoming = canonicalToDir(cylindrical_direction);
#endif

            const float product = (*m_parent_map)[index] * bsdf_proxy.evaluate(incoming);
            m_map[index] = product;
            distribution_sum += product;
            distribution[index] = distribution_sum;
        }
    }
    // Build discrete 2D distribution
    m_image_importance_sampler.build();
}

void RadianceProxy::build_product_simd(
    BSDFProxy &bsdf_proxy,
    const Vector3f &outgoing,
    const Vector3f &shading_normal)
{
    assert(m_is_built);

    if (m_product_is_built)
        return;

    bsdf_proxy.finish_parameterization(outgoing, shading_normal);
    m_product_is_built = true;

    const Vec8f inv_width(1.0f / ProxyWidth);
    const Vec8i simd_loop_offsets(0, 1, 2, 3, 4, 5, 6, 7);

    float* distribution = m_image_importance_sampler.get_distribution();

    Vec8f distribution_sum_left(Zero_SIMD);
    Vec8f distribution_sum_right(Zero_SIMD);

    if (!cosines_loaded)
    {
        std::lock_guard<std::mutex> guard(cosine_mutex);

        if(!cosines_loaded)
        {
            char* data = reinterpret_cast<char*>(cosines);
            std::ifstream file("/mnt/Data/_Programming/PracticalPathGuiding/cosines.bin", std::ios::in | std::ios::binary);
            file.read(data, ProxyWidth * ProxyWidth * ProxyWidth * ProxyWidth * sizeof(float));
            file.close();

            cosines_loaded = true;
        }
    }

    Vector3f diffuse_lobe, translucency_lobe, reflectance_lobe, refractance_lobe;
    bsdf_proxy.get_lobes(diffuse_lobe, translucency_lobe, reflectance_lobe, refractance_lobe);

    Point2f diffuse_scaled = dirToCanonical(diffuse_lobe) * static_cast<float>(ProxyWidth);
    Point2f reflectance_scaled = dirToCanonical(reflectance_lobe) * static_cast<float>(ProxyWidth);

    Point2u diffuse_pixel(diffuse_scaled.x, diffuse_scaled.y);
    Point2u reflectance_pixel(reflectance_scaled.x, reflectance_scaled.y);

    diffuse_pixel.x = std::min(static_cast<size_t>(diffuse_pixel.x), ProxyWidth - 1);
    diffuse_pixel.y = std::min(static_cast<size_t>(diffuse_pixel.y), ProxyWidth - 1);
    reflectance_pixel.x = std::min(static_cast<size_t>(reflectance_pixel.x), ProxyWidth - 1);
    reflectance_pixel.y = std::min(static_cast<size_t>(reflectance_pixel.y), ProxyWidth - 1);

    const float* diffuse_cosines_array = &cosines[(diffuse_pixel.y * ProxyWidth + diffuse_pixel.x) * ProxyWidth * ProxyWidth];
    const float* reflectance_cosines_array = &cosines[(reflectance_pixel.y * ProxyWidth + reflectance_pixel.x) * ProxyWidth * ProxyWidth];

    for (size_t y = 0; y < ProxyWidth; ++y)
    {
        const Vec8i pixel_y(y);

        for (size_t x = 0; x < ProxyWidth; x += 8)
        {
            const Vec8i pixel_x = Vec8i(x) + simd_loop_offsets;

            const size_t index = y * ProxyWidth + x;


#ifdef LOAD_INCOMING_ARRAY
            const Vec8f &incoming_x = *reinterpret_cast<const Vec8f*>(worldDir_soa + (index * 3));
            const Vec8f &incoming_y = *reinterpret_cast<const Vec8f*>(worldDir_soa + (index * 3 + 8));
            const Vec8f &incoming_z = *reinterpret_cast<const Vec8f*>(worldDir_soa + (index * 3 + 2 * 8));
            // incoming_x.load_a(worldDir_soa + (index * 3));
            // incoming_y.load_a(worldDir_soa + (index * 3 + 8));
            // incoming_z.load_a(worldDir_soa + (index * 3 + 2 * 8));
#else
            const Vec8f cylindrical_direction_x = (to_float(pixel_x) + Vec8f(0.5f)) * inv_width;
            const Vec8f cylindrical_direction_y = (to_float(pixel_y) + Vec8f(0.5f)) * inv_width;
            canonicalToDir_simd(cylindrical_direction_x, cylindrical_direction_y, incoming_x, incoming_y, incoming_z);
#endif
            Vec8f diffuse_cosines, reflectance_cosines;
            diffuse_cosines.load_a(&diffuse_cosines_array[index]);
            reflectance_cosines.load_a(&reflectance_cosines_array[index]);

            const Vec8f bsdf_proxy_value = bsdf_proxy.evaluate_simd(incoming_x, incoming_y, incoming_z, diffuse_cosines, reflectance_cosines);
            Vec8f radiance;
            radiance.load(&((*m_parent_map)[index]));
            radiance *= bsdf_proxy_value;
            radiance.store(&m_map[index]);

            if (x == 0)
            {
                distribution_sum_left += select(radiance > Zero_SIMD, radiance, Zero_SIMD);
                distribution_sum_left.store(&distribution[index]);
            }
            else
            {
                distribution_sum_right += select(radiance > Zero_SIMD, radiance, Zero_SIMD);
                distribution_sum_right.store(&distribution[index]);
            }
        }
    }

    // float cosines[ProxyWidth * ProxyWidth * ProxyWidth * ProxyWidth];

    // for (size_t dest_y = 0; dest_y < ProxyWidth; ++dest_y)
    // {
    //     for (size_t dest_x = 0; dest_x < ProxyWidth; ++dest_x)
    //     {
    //         const Point2u dest_pixel(dest_x, dest_y);
    //         size_t dest_index = dest_y * ProxyWidth + dest_x;

    //         for (size_t y = 0; y < ProxyWidth; ++y)
    //         {
    //             for (size_t x = 0; x < ProxyWidth; ++x)
    //             {
    //                 const Point2u incoming_pixel(x, y);
    //                 size_t index = y * ProxyWidth + x;

    //                 const float cos_theta = cos_theta_to_boundary(incoming_pixel, dest_pixel, ProxyWidth);
    //                 cosines[dest_index * ProxyWidth * ProxyWidth + index] = cos_theta;

    //                 if (cos_theta > M_PI / 4.0f && cos_theta != 1.0f)
    //                 {
    //                     const float cos_theta_2 = cos_theta_to_boundary(incoming_pixel, dest_pixel, ProxyWidth);
    //                 }
    //             }
    //         }
    //     }
    // }

    // const char* data = reinterpret_cast<const char*>(cosines);
    // std::ofstream file("/mnt/Data/_Programming/PracticalPathGuiding/cosines.bin", std::ios::out | std::ios::binary);
    // file.write(data, ProxyWidth * ProxyWidth * ProxyWidth * ProxyWidth * sizeof(float));
    // file.close();

    m_image_importance_sampler.build();
}

void RadianceProxy::clear()
{
    m_is_built = false;
    m_product_is_built = false;
    m_quadtree_nodes = nullptr;
}

float RadianceProxy::proxy_radiance(
    const Vector3f &direction) const
{
    const Point2f spherical_direction(dirToCanonical(direction) * static_cast<float>(ProxyWidth));
    const Point2u pixel(
        std::min(static_cast<size_t>(spherical_direction.x), ProxyWidth - 1),
        std::min(static_cast<size_t>(spherical_direction.y), ProxyWidth - 1));

    return m_map[pixel.y * ProxyWidth + pixel.x];
}

float RadianceProxy::sample(
    Sampler *sampler,
    Vector3f &direction) const
{
    assert(m_is_built);
    assert(m_product_is_built);
    // Sample the importance map.
    Point2f s = sampler->next2D();
    Point2u pixel;

    float pdf;
    m_image_importance_sampler.sample(s, pixel);

    if (!m_image_importance_sampler.is_zero())
    {
        const size_t map_index = pixel.y * ProxyWidth + pixel.x;
        pdf = m_map[map_index] / m_image_importance_sampler.get_total_radiance();
    }
    else
    {
        pdf = 1.0f / (ProxyWidth * ProxyWidth);
    }

    assert(pdf >= 0.0f);

    Point2f cylindrical_direction(pixel.x, pixel.y);

    assert(m_quadtree_strata != nullptr);
    assert(m_quadtree_strata->size() == ProxyWidth * ProxyWidth);
    assert(pixel.y * ProxyWidth + pixel.x < ProxyWidth * ProxyWidth && pixel.y * ProxyWidth + pixel.x >= 0);
    const QuadTreeNode *sub_tree = (*m_quadtree_strata)[pixel.y * ProxyWidth + pixel.x];

    if (sub_tree)
    {
        assert(m_quadtree_nodes != nullptr);
        const Point2 sub_direction = sub_tree->sample(sampler, *m_quadtree_nodes);
        Point2 sub_direction_pdf_eval = sub_direction;
        const float tree_pdf = sub_tree->pdf(sub_direction_pdf_eval, *m_quadtree_nodes);
        cylindrical_direction += sub_direction;
        pdf *= tree_pdf;
    }
    else
    {
        s = sampler->next2D();
        cylindrical_direction += s;
    }

    pdf *= ProxyWidth * ProxyWidth * INV_FOURPI;
    cylindrical_direction *= 1.0f / ProxyWidth;
    // assert(cylindrical_direction.x >= 0.0f && cylindrical_direction.x < 1.0f);
    // assert(cylindrical_direction.y >= 0.0f && cylindrical_direction.y < 1.0f);
    cylindrical_direction.x = std::max(std::min(cylindrical_direction.x, 0.99999f), 0.0f);
    cylindrical_direction.y = std::max(std::min(cylindrical_direction.y, 0.99999f), 0.0f);
    direction = canonicalToDir(cylindrical_direction);

    return pdf;
}

float RadianceProxy::pdf(
    const Vector3f &direction) const
{
    assert(m_is_built);
    assert(m_product_is_built);

    const Point2f cylindrical_direction = dirToCanonical(direction);
    const Point2f cylindrical_direction_scaled = cylindrical_direction * static_cast<float>(ProxyWidth);
    Point2u pixel(cylindrical_direction_scaled.x, cylindrical_direction_scaled.y);

    pixel.x = std::min(pixel.x, static_cast<unsigned int>(ProxyWidth - 1));
    pixel.y = std::min(pixel.y, static_cast<unsigned int>(ProxyWidth - 1));

    // TODO: More precise mapping between directions and map pixels to avoid discrepancies in sampled
    // and evaluated pdf values. There also seems to be another source causing these discrepancies.

    float pdf;

    if (!m_image_importance_sampler.is_zero())
    {
        const size_t map_index = pixel.y * ProxyWidth + pixel.x;
        pdf = m_map[map_index] / m_image_importance_sampler.get_total_radiance();
    }
    else
    {
        pdf = 1.0f / (ProxyWidth * ProxyWidth);
    }


    assert(m_quadtree_strata != nullptr);
    assert(m_quadtree_strata->size() == ProxyWidth * ProxyWidth);
    assert(pixel.y * ProxyWidth + pixel.x >= 0);
    assert(pixel.y * ProxyWidth + pixel.x < ProxyWidth * ProxyWidth);
    const QuadTreeNode *sub_tree = (*m_quadtree_strata)[pixel.y * ProxyWidth + pixel.x];

    if (sub_tree)
    {
        assert(m_quadtree_nodes != nullptr);
        Point2f sub_direction(cylindrical_direction_scaled.x - pixel.x, cylindrical_direction_scaled.y - pixel.y);
        pdf *= sub_tree->pdf(sub_direction, *m_quadtree_nodes);
    }

    pdf *= ProxyWidth * ProxyWidth * INV_FOURPI;
    return pdf;
}

bool RadianceProxy::is_built() const
{
    return m_is_built;
}

void RadianceProxy::set_maps(const RadianceProxy &other)
{
    m_parent_map = &other.m_map;
    m_quadtree_strata = other.m_quadtree_strata;
    m_quadtree_nodes = other.m_quadtree_nodes;
    m_is_built = other.m_is_built;
}

class DTree {
public:
    DTree() {
        m_atomic.sum.store(0, std::memory_order_relaxed);
        m_maxDepth = 0;
        m_nodes.emplace_back();
        m_nodes.front().setSum(0.0f);
    }

    const QuadTreeNode& node(size_t i) const {
        return m_nodes[i];
    }

    Float mean() const {
        if (m_atomic.statisticalWeight == 0) {
            return 0;
        }
        const Float factor = 1 / (M_PI * 4 * m_atomic.statisticalWeight);
        return factor * m_atomic.sum;
    }

    Spectrum getMeasurement() const {
        return m_atomic.getMeasurement();
    }

    void recordIrradiance(Point2 p, Float irradiance, Float statisticalWeight, EDirectionalFilter directionalFilter) {
        if (std::isfinite(statisticalWeight) && statisticalWeight > 0) {
            addToAtomicFloat(m_atomic.statisticalWeight, statisticalWeight);

            if (std::isfinite(irradiance) && irradiance > 0) {
                if (directionalFilter == EDirectionalFilter::ENearest) {
                    m_nodes[0].record(p, irradiance * statisticalWeight, m_nodes);
                } else {
                    int depth = depthAt(p);
                    Float size = std::pow(0.5f, depth);

                    Point2 origin = p;
                    origin.x -= size / 2;
                    origin.y -= size / 2;
                    m_nodes[0].record(origin, size, Point2(0.0f), 1.0f, irradiance * statisticalWeight / (size * size), m_nodes);
                }
            }
        }
    }

    void recordMeasurement(const Spectrum &m)
    {
        m_atomic.recordMeasurement(m);
    }

    Float eval(Point2 p) const {
        if (m_atomic.statisticalWeight == 0.0f)
        {
            return 0.0f;
        }

        return m_nodes[0].eval(p, m_nodes) / (4.0f * M_PI * m_atomic.statisticalWeight);
    }

    Float pdf(Point2 p) const {
        if (!(mean() > 0)) {
            return 1 / (4 * M_PI);
        }

        return m_nodes[0].pdf(p, m_nodes) / (4 * M_PI);
    }

    int depthAt(Point2 p) const {
        return m_nodes[0].depthAt(p, m_nodes);
    }

    int depth() const {
        return m_maxDepth;
    }

    Point2 sample(Sampler* sampler) const {
        if (!(mean() > 0)) {
            return sampler->next2D();
        }

        Point2 res = m_nodes[0].sample(sampler, m_nodes);

        res.x = math::clamp(res.x, 0.0f, 1.0f);
        res.y = math::clamp(res.y, 0.0f, 1.0f);

        return res;
    }

    size_t numNodes() const {
        return m_nodes.size();
    }

    Float statisticalWeight() const {
        return m_atomic.statisticalWeight;
    }

    void setStatisticalWeight(Float statisticalWeight) {
        m_atomic.statisticalWeight = statisticalWeight;
    }

    void reset(const DTree& previousDTree, int newMaxDepth, Float subdivisionThreshold) {
        m_atomic = Atomic{};
        m_maxDepth = 0;
        m_nodes.clear();
        m_nodes.emplace_back();

        struct StackNode {
            size_t nodeIndex;
            size_t otherNodeIndex;
            const DTree* otherDTree;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0, 0, &previousDTree, 1});

        const Float total = previousDTree.m_atomic.sum;
        
        // Create the topology of the new DTree to be the refined version
        // of the previous DTree. Subdivision is recursive if enough energy is there.
        while (!nodeIndices.empty()) {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            m_maxDepth = std::max(m_maxDepth, sNode.depth);

            for (int i = 0; i < 4; ++i) {
                const QuadTreeNode& otherNode = sNode.otherDTree->m_nodes[sNode.otherNodeIndex];
                const Float fraction = total > 0 ? (otherNode.sum(i) / total) : std::pow(0.25f, sNode.depth);
                SAssert(fraction <= 1.0f + Epsilon);

                if (sNode.depth < newMaxDepth && fraction > subdivisionThreshold) {
                    if (!otherNode.isLeaf(i)) {
                        SAssert(sNode.otherDTree == &previousDTree);
                        nodeIndices.push({m_nodes.size(), otherNode.child(i), &previousDTree, sNode.depth + 1});
                    } else {
                        nodeIndices.push({m_nodes.size(), m_nodes.size(), this, sNode.depth + 1});
                    }

                    m_nodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(m_nodes.size()));
                    m_nodes.emplace_back();
                    m_nodes.back().setSum(otherNode.sum(i) / 4);

                    if (m_nodes.size() > std::numeric_limits<uint16_t>::max()) {
                        SLog(EWarn, "DTreeWrapper hit maximum children count.");
                        nodeIndices = std::stack<StackNode>();
                        break;
                    }
                }
            }
        }

        // Uncomment once memory becomes an issue.
        //m_nodes.shrink_to_fit();

        for (auto& node : m_nodes) {
            node.setSum(0);
        }
    }

    size_t approxMemoryFootprint() const {
        return m_nodes.capacity() * sizeof(QuadTreeNode) + sizeof(*this);
    }

    void build() {
        auto& root = m_nodes[0];

        // Build the quadtree recursively, starting from its root.
        root.build(m_nodes);

        // Ensure that the overall sum of irradiance estimates equals
        // the sum of irradiance estimates found in the quadtree.
        Float sum = 0;
        for (int i = 0; i < 4; ++i) {
            sum += root.sum(i);
        }
        m_atomic.sum.store(sum);
    }

    void buildRadianceProxy(RadianceProxy& radianceProxy)
    {
        const float statisticalWeight = m_atomic.statisticalWeight.load();
        const float sum = m_atomic.sum.load();

        if (sum > 0.0f && statisticalWeight > 0.0f)
            radianceProxy.build(&m_nodes, INV_FOURPI / statisticalWeight);
        else
            radianceProxy.clear();
    }

    float radiance(Vector& dir) const
    {
        if (m_atomic.sum.load() <= 0.0f || m_atomic.statisticalWeight.load() <= 0.0)
            return 0.0f;

        Point2 cylindrical_direction = dirToCanonical(dir);
        return m_nodes[0].radiance(cylindrical_direction, m_nodes) / (4.0f * M_PI * m_atomic.statisticalWeight.load());
    }

private:
    std::vector<QuadTreeNode> m_nodes;

    struct Atomic {
        Atomic() {
            sum.store(0, std::memory_order_relaxed);
            statisticalWeight.store(0, std::memory_order_relaxed);

            measurementR.store(0, std::memory_order_relaxed);
            measurementG.store(0, std::memory_order_relaxed);
            measurementB.store(0, std::memory_order_relaxed);
            nMeasurementSamples = 0;
        }

        Atomic(const Atomic& arg) {
            *this = arg;
        }

        Atomic &operator=(const Atomic &arg)
        {
            sum.store(arg.sum.load(std::memory_order_relaxed), std::memory_order_relaxed);
            statisticalWeight.store(arg.statisticalWeight.load(std::memory_order_relaxed), std::memory_order_relaxed);

            measurementR.store(arg.measurementR.load(std::memory_order_relaxed), std::memory_order_relaxed);
            measurementG.store(arg.measurementG.load(std::memory_order_relaxed), std::memory_order_relaxed);
            measurementB.store(arg.measurementB.load(std::memory_order_relaxed), std::memory_order_relaxed);
            nMeasurementSamples = arg.nMeasurementSamples.load(std::memory_order_relaxed);

            return *this;
        }

        std::atomic<Float> sum;
        std::atomic<Float> statisticalWeight;

        void recordMeasurement(const Spectrum &val)
        {
            ++nMeasurementSamples;
            addToAtomicFloat(measurementR, val[0]);
            addToAtomicFloat(measurementG, val[1]);
            addToAtomicFloat(measurementB, val[2]);
        }

        Spectrum getMeasurement() const
        {
            size_t n = nMeasurementSamples.load(std::memory_order_relaxed);
            if (n == 0)
            {
                return Spectrum{0.0f};
            }

            Spectrum result;
            result[0] = measurementR.load(std::memory_order_relaxed);
            result[1] = measurementG.load(std::memory_order_relaxed);
            result[2] = measurementB.load(std::memory_order_relaxed);
            return result / (Float)n;
        }

        std::atomic<Float> measurementR;
        std::atomic<Float> measurementG;
        std::atomic<Float> measurementB;
        std::atomic<size_t> nMeasurementSamples;

    } m_atomic;

    int m_maxDepth;
};

struct DTreeRecord {
    Vector d;
    Float radiance, product;
    Float woPdf, bsdfPdf, dTreePdf, productPdf;
    EGuidingMode guidingMode;
    Float statisticalWeight;
    bool isDelta;
    EGuidingMode bounceMode;
};

struct DTreeWrapper {
public:
    DTreeWrapper() {
    }

    void record(const DTreeRecord& rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss) {
        if (!rec.isDelta) {
            Float irradiance = rec.radiance / rec.woPdf;
            building.recordIrradiance(dirToCanonical(rec.d), irradiance, rec.statisticalWeight, directionalFilter);
        }

        if (bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone && rec.product > 0) {
            if (rec.guidingMode == EGuidingMode::ECombined)
                optimizeBsdfSamplingFractionCombined(rec);
            else
            {
                if (rec.bounceMode == EGuidingMode::ECombined)
                {
                    optimizeBsdfSamplingFraction(rec, bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EKL ? 1.0f : 2.0f, bsdfSamplingFractionOptimizerProduct);
                    optimizeBsdfSamplingFraction(rec, bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EKL ? 1.0f : 2.0f, bsdfSamplingFractionOptimizerRadiance);
                }
                else if (rec.bounceMode == EGuidingMode::EProduct)
                {
                    optimizeBsdfSamplingFraction(rec, bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EKL ? 1.0f : 2.0f, bsdfSamplingFractionOptimizerProduct);
                }
                if (rec.bounceMode == EGuidingMode::ERadiance)
                {
                    optimizeBsdfSamplingFraction(rec, bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EKL ? 1.0f : 2.0f, bsdfSamplingFractionOptimizerRadiance);
                }
            }
        }
    }

    void recordMeasurement(Spectrum m)
    {
        if (!m.isValid())
        {
            m = Spectrum{0.0f};
        }
        building.recordMeasurement(m);
    }

    Float estimateRadiance(Point2 p) const
    {
        return sampling.eval(p);
    }

    Float estimateRadiance(const Vector &d) const
    {
        return estimateRadiance(dirToCanonical(d));
    }

    void build() {
        building.build();
        sampling = building;
    }

    void reset(int maxDepth, Float subdivisionThreshold) {
        building.reset(sampling, maxDepth, subdivisionThreshold);
        sampling.buildRadianceProxy(radianceProxy);
    }

    Vector sample(Sampler* sampler) const {
        return canonicalToDir(sampling.sample(sampler));
    }

    Float pdf(const Vector& dir) const {
        return sampling.pdf(dirToCanonical(dir));
    }

    Float diff(const DTreeWrapper& other) const {
        return 0.0f;
    }

    int depth() const {
        return sampling.depth();
    }

    size_t numNodes() const {
        return sampling.numNodes();
    }

    Float meanRadiance() const {
        return sampling.mean();
    }

    Spectrum measurementEstimate() const {
        return sampling.getMeasurement();
    }

    Float statisticalWeight() const {
        return sampling.statisticalWeight();
    }

    Float statisticalWeightBuilding() const {
        return building.statisticalWeight();
    }

    void setStatisticalWeightBuilding(Float statisticalWeight) {
        building.setStatisticalWeight(statisticalWeight);
    }

    size_t approxMemoryFootprint() const {
        return building.approxMemoryFootprint() + sampling.approxMemoryFootprint();
    }

    inline Float bsdfSamplingFraction(Float variable) const {
        return logistic(variable);
    }

    inline Float dBsdfSamplingFraction_dVariable(Float variable) const {
        Float fraction = bsdfSamplingFraction(variable);
        return fraction * (1 - fraction);
    }

    inline Float bsdfSamplingFractionProduct() const {
        return bsdfSamplingFraction(bsdfSamplingFractionOptimizerProduct.variable());
    }

    inline Float bsdfSamplingFractionRadiance() const {
        return bsdfSamplingFraction(bsdfSamplingFractionOptimizerRadiance.variable());
    }

    inline Vector2 bsdfSamplingFractionCombined() const {
        m_lock_product.lock();
        const Vector2 variableCombined = bsdfSamplingFractionOptimizerProduct.variable_combined();
        m_lock_product.unlock();
        return Vector2(
            bsdfSamplingFraction(variableCombined.x),
            bsdfSamplingFraction(variableCombined.y));
    }

    void optimizeBsdfSamplingFraction(const DTreeRecord& rec, Float ratioPower, AdamOptimizer& optimizer) {
        m_lock.lock();

        // GRADIENT COMPUTATION
        Float variable = optimizer.variable();
        Float samplingFraction = bsdfSamplingFraction(variable);

        // Loss gradient w.r.t. sampling fraction
        const Float guidedPdf = rec.bounceMode == EGuidingMode::EProduct ? rec.productPdf : rec.dTreePdf;
        Float mixPdf = samplingFraction * rec.bsdfPdf + (1 - samplingFraction) * guidedPdf;
        Float ratio = std::pow(rec.product / mixPdf, ratioPower);
        Float dLoss_dSamplingFraction = -ratio / rec.woPdf * (rec.bsdfPdf - guidedPdf);

        // Chain rule to get loss gradient w.r.t. trainable variable
        Float dLoss_dVariable = dLoss_dSamplingFraction * dBsdfSamplingFraction_dVariable(variable);

        // We want some regularization such that our parameter does not become too big.
        // We use l2 regularization, resulting in the following linear gradient.
        Float l2RegGradient = 0.01f * variable;

        Float lossGradient = l2RegGradient + dLoss_dVariable;

        // ADAM GRADIENT DESCENT
        optimizer.append(lossGradient, rec.statisticalWeight);

        m_lock.unlock();
    }

    void optimizeBsdfSamplingFractionCombined(const DTreeRecord& rec) {
        m_lock_product.lock();

        // GRADIENT COMPUTATION
        Vector2 variable = bsdfSamplingFractionOptimizerProduct.variable_combined();
        Vector2 samplingFraction(bsdfSamplingFraction(variable.x), bsdfSamplingFraction(variable.y));

        // Loss gradient w.r.t. sampling fraction
        Float mixPdf = samplingFraction.x * rec.bsdfPdf +
                    (1.0f - samplingFraction.x) * (samplingFraction.y * rec.productPdf +
                    (1.0f - samplingFraction.y) * rec.dTreePdf);

        Vector2 dSamplingFraction(-rec.product / (rec.woPdf * mixPdf));
        dSamplingFraction.x *= rec.bsdfPdf - samplingFraction.y * rec.productPdf +
                                              (1.0f - samplingFraction.y) * rec.dTreePdf;
        dSamplingFraction.y *= (1.0f - samplingFraction.x) * (rec.productPdf - rec.dTreePdf);

        Vector2 d_theta(
            dSamplingFraction.x * samplingFraction.x * (1.0f - samplingFraction.x),
            dSamplingFraction.y * samplingFraction.y * (1.0f - samplingFraction.y));

        // We want some regularization such that our parameter does not become too big.
        // We use l2 regularization, resulting in the following linear gradient.
        Vector2 l2RegGradient = 0.01f * variable;

        Vector2 lossGradient = l2RegGradient + d_theta;

        // ADAM GRADIENT DESCENT
        bsdfSamplingFractionOptimizerProduct.append_product(lossGradient, rec.statisticalWeight);

        m_lock_product.unlock();
    }

    void dump(BlobWriter& blob, const Point& p, const Vector& size) const {
        blob
            << (float)p.x << (float)p.y << (float)p.z
            << (float)size.x << (float)size.y << (float)size.z
            << (float)sampling.mean() << (uint64_t)sampling.statisticalWeight() << (uint64_t)sampling.numNodes();

        for (size_t i = 0; i < sampling.numNodes(); ++i) {
            const auto& node = sampling.node(i);
            for (int j = 0; j < 4; ++j) {
                blob << (float)node.sum(j) << (uint16_t)node.child(j);
            }
        }
    }

    const RadianceProxy& getRadianceProxy() const
    {
        return radianceProxy;
    }

    float radiance(Vector& dir) const
    {
        return sampling.radiance(dir);
    }

private:
    DTree building;
    DTree sampling;

    AdamOptimizer bsdfSamplingFractionOptimizerProduct{0.01, 0.001f};
    AdamOptimizer bsdfSamplingFractionOptimizerRadiance{0.01, 0.001f};

    RadianceProxy radianceProxy;

    class SpinLock {
    public:
        SpinLock() {
            m_mutex.clear(std::memory_order_release);
        }

        SpinLock(const SpinLock& other) { m_mutex.clear(std::memory_order_release); }
        SpinLock& operator=(const SpinLock& other) { return *this; }

        void lock() {
            while (m_mutex.test_and_set(std::memory_order_acquire)) { }
        }

        void unlock() {
            m_mutex.clear(std::memory_order_release);
        }
    private:
        std::atomic_flag m_mutex;
    };

    SpinLock m_lock;
    mutable SpinLock m_lock_product;
};

struct STreeNode {
    STreeNode() {
        children = {};
        isLeaf = true;
        axis = 0;
    }

    int childIndex(Point& p) const {
        if (p[axis] < 0.5f) {
            p[axis] *= 2;
            return 0;
        } else {
            p[axis] = (p[axis] - 0.5f) * 2;
            return 1;
        }
    }

    int nodeIndex(Point& p) const {
        return children[childIndex(p)];
    }

    DTreeWrapper* dTreeWrapper(Point& p, Vector& size, std::vector<STreeNode>& nodes) {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return &dTree;
        } else {
            size[axis] /= 2;
            return nodes[nodeIndex(p)].dTreeWrapper(p, size, nodes);
        }
    }

    const DTreeWrapper* dTreeWrapper() const {
        return &dTree;
    }

    int depth(Point& p, const std::vector<STreeNode>& nodes) const {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf) {
            return 1;
        } else {
            return 1 + nodes[nodeIndex(p)].depth(p, nodes);
        }
    }

    int depth(const std::vector<STreeNode>& nodes) const {
        int result = 1;

        if (!isLeaf) {
            for (auto c : children) {
                result = std::max(result, 1 + nodes[c].depth(nodes));
            }
        }

        return result;
    }

    void forEachLeaf(
        std::function<void(const DTreeWrapper*, const Point&, const Vector&)> func,
        Point p, Vector size, const std::vector<STreeNode>& nodes) const {

        if (isLeaf) {
            func(&dTree, p, size);
        } else {
            size[axis] /= 2;
            for (int i = 0; i < 2; ++i) {
                Point childP = p;
                if (i == 1) {
                    childP[axis] += size[axis];
                }

                nodes[children[i]].forEachLeaf(func, childP, size, nodes);
            }
        }
    }

    Float computeOverlappingVolume(const Point& min1, const Point& max1, const Point& min2, const Point& max2) {
        Float lengths[3];
        for (int i = 0; i < 3; ++i) {
            lengths[i] = std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1] * lengths[2];
    }

    void record(const Point& min1, const Point& max1, Point min2, Vector size2, const DTreeRecord& rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, std::vector<STreeNode>& nodes) {
        Float w = computeOverlappingVolume(min1, max1, min2, min2 + size2);
        if (w > 0) {
            if (isLeaf) {
                dTree.record({ rec.d, rec.radiance, rec.product, rec.woPdf, rec.bsdfPdf, rec.dTreePdf, rec.productPdf, rec.guidingMode, rec.statisticalWeight * w, rec.isDelta, rec.bounceMode}, directionalFilter, bsdfSamplingFractionLoss);
            } else {
                size2[axis] /= 2;
                for (int i = 0; i < 2; ++i) {
                    if (i & 1) {
                        min2[axis] += size2[axis];
                    }

                    nodes[children[i]].record(min1, max1, min2, size2, rec, directionalFilter, bsdfSamplingFractionLoss, nodes);
                }
            }
        }
    }

    bool isLeaf;
    DTreeWrapper dTree;
    int axis;
    std::array<uint32_t, 2> children;
};


class STree {
public:
    STree(const AABB& aabb) {
        clear();

        m_aabb = aabb;

        // Enlarge AABB to turn it into a cube. This has the effect
        // of nicer hierarchical subdivisions.
        Vector size = m_aabb.max - m_aabb.min;
        Float maxSize = std::max(std::max(size.x, size.y), size.z);
        m_aabb.max = m_aabb.min + Vector(maxSize);
    }

    void clear() {
        m_nodes.clear();
        m_nodes.emplace_back();
    }

    void subdivideAll() {
        int nNodes = (int)m_nodes.size();
        for (int i = 0; i < nNodes; ++i) {
            if (m_nodes[i].isLeaf) {
                subdivide(i, m_nodes);
            }
        }
    }

    void subdivide(int nodeIdx, std::vector<STreeNode>& nodes) {
        // Add 2 child nodes
        nodes.resize(nodes.size() + 2);

        if (nodes.size() > std::numeric_limits<uint32_t>::max()) {
            SLog(EWarn, "DTreeWrapper hit maximum children count.");
            return;
        }

        STreeNode& cur = nodes[nodeIdx];
        for (int i = 0; i < 2; ++i) {
            uint32_t idx = (uint32_t)nodes.size() - 2 + i;
            cur.children[i] = idx;
            nodes[idx].axis = (cur.axis + 1) % 3;
            nodes[idx].dTree = cur.dTree;
            nodes[idx].dTree.setStatisticalWeightBuilding(nodes[idx].dTree.statisticalWeightBuilding() / 2);
        }
        cur.isLeaf = false;
        cur.dTree = {}; // Reset to an empty dtree to save memory.
    }

    DTreeWrapper* dTreeWrapper(Point p, Vector& size) {
        size = m_aabb.getExtents();
        p = Point(p - m_aabb.min);
        p.x /= size.x;
        p.y /= size.y;
        p.z /= size.z;

        return m_nodes[0].dTreeWrapper(p, size, m_nodes);
    }

    DTreeWrapper* dTreeWrapper(Point p) {
        Vector size;
        return dTreeWrapper(p, size);
    }

    void forEachDTreeWrapperConst(std::function<void(const DTreeWrapper*)> func) const {
        for (auto& node : m_nodes) {
            if (node.isLeaf) {
                func(&node.dTree);
            }
        }
    }

    void forEachDTreeWrapperConstP(std::function<void(const DTreeWrapper*, const Point&, const Vector&)> func) const {
        m_nodes[0].forEachLeaf(func, m_aabb.min, m_aabb.max - m_aabb.min, m_nodes);
    }

    void forEachDTreeWrapperParallel(std::function<void(DTreeWrapper*)> func) {
        int nDTreeWrappers = static_cast<int>(m_nodes.size());

#pragma omp parallel for
        for (int i = 0; i < nDTreeWrappers; ++i) {
            if (m_nodes[i].isLeaf) {
                func(&m_nodes[i].dTree);
            }
        }
    }

    void record(const Point& p, const Vector& dTreeVoxelSize, DTreeRecord rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss) {
        Float volume = 1;
        for (int i = 0; i < 3; ++i) {
            volume *= dTreeVoxelSize[i];
        }

        rec.statisticalWeight /= volume;
        m_nodes[0].record(p - dTreeVoxelSize * 0.5f, p + dTreeVoxelSize * 0.5f, m_aabb.min, m_aabb.getExtents(), rec, directionalFilter, bsdfSamplingFractionLoss, m_nodes);
    }

    void dump(BlobWriter& blob) const {
        forEachDTreeWrapperConstP([&blob](const DTreeWrapper* dTree, const Point& p, const Vector& size) {
            if (dTree->statisticalWeight() > 0) {
                dTree->dump(blob, p, size);
            }
        });
    }

    bool shallSplit(const STreeNode& node, int depth, size_t samplesRequired) {
        return m_nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.statisticalWeightBuilding() > samplesRequired;
    }

    void refine(size_t sTreeThreshold, int maxMB) {
        if (maxMB >= 0) {
            size_t approxMemoryFootprint = 0;
            for (const auto& node : m_nodes) {
                approxMemoryFootprint += node.dTreeWrapper()->approxMemoryFootprint();
            }

            if (approxMemoryFootprint / 1000000 >= (size_t)maxMB) {
                return;
            }
        }
        
        struct StackNode {
            size_t index;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0,  1});
        while (!nodeIndices.empty()) {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            // Subdivide if needed and leaf
            if (m_nodes[sNode.index].isLeaf) {
                if (shallSplit(m_nodes[sNode.index], sNode.depth, sTreeThreshold)) {
                    subdivide((int)sNode.index, m_nodes);
                }
            }

            // Add children to stack if we're not
            if (!m_nodes[sNode.index].isLeaf) {
                const STreeNode& node = m_nodes[sNode.index];
                for (int i = 0; i < 2; ++i) {
                    nodeIndices.push({node.children[i], sNode.depth + 1});
                }
            }
        }

        // Uncomment once memory becomes an issue.
        //m_nodes.shrink_to_fit();
    }

    const AABB& aabb() const {
        return m_aabb;
    }

private:
    std::vector<STreeNode> m_nodes;
    AABB m_aabb;
};


static StatsCounter avgPathLength("Guided path tracer", "Average path length", EAverage);

class GuidedPathTracer : public MonteCarloIntegrator {
public:
    GuidedPathTracer(const Properties &props) : MonteCarloIntegrator(props) {
        m_neeStr = props.getString("nee", "never");
        if (m_neeStr == "never") {
            m_nee = ENever;
        } else if (m_neeStr == "kickstart") {
            m_nee = EKickstart;
        } else if (m_neeStr == "always") {
            m_nee = EAlways;
        } else {
            Assert(false);
        }

        m_sampleCombinationStr = props.getString("sampleCombination", "automatic");
        if (m_sampleCombinationStr == "discard") {
            m_sampleCombination = ESampleCombination::EDiscard;
        } else if (m_sampleCombinationStr == "automatic") {
            m_sampleCombination = ESampleCombination::EDiscardWithAutomaticBudget;
        } else if (m_sampleCombinationStr == "inversevar") {
            m_sampleCombination = ESampleCombination::EInverseVariance;
        } else {
            Assert(false);
        }

        m_spatialFilterStr = props.getString("spatialFilter", "nearest");
        if (m_spatialFilterStr == "nearest") {
            m_spatialFilter = ESpatialFilter::ENearest;
        } else if (m_spatialFilterStr == "stochastic") {
            m_spatialFilter = ESpatialFilter::EStochasticBox;
        } else if (m_spatialFilterStr == "box") {
            m_spatialFilter = ESpatialFilter::EBox;
        } else {
            Assert(false);
        }

        m_directionalFilterStr = props.getString("directionalFilter", "nearest");
        if (m_directionalFilterStr == "nearest") {
            m_directionalFilter = EDirectionalFilter::ENearest;
        } else if (m_directionalFilterStr == "box") {
            m_directionalFilter = EDirectionalFilter::EBox;
        } else {
            Assert(false);
        }

        m_bsdfSamplingFractionLossStr = props.getString("bsdfSamplingFractionLoss", "none");
        if (m_bsdfSamplingFractionLossStr == "none") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::ENone;
        } else if (m_bsdfSamplingFractionLossStr == "kl") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EKL;
        } else if (m_bsdfSamplingFractionLossStr == "var") {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EVariance;
        } else {
            Assert(false);
        }

        m_sdTreeMaxMemory = props.getInteger("sdTreeMaxMemory", -1);
        m_sTreeThreshold = props.getInteger("sTreeThreshold", 12000);
        m_dTreeThreshold = props.getFloat("dTreeThreshold", 0.01f);
        m_bsdfSamplingFraction = props.getFloat("bsdfSamplingFraction", 0.5f);
        m_sppPerPass = props.getInteger("sppPerPass", 4);

        m_budgetStr = props.getString("budgetType", "seconds");
        if (m_budgetStr == "spp") {
            m_budgetType = ESpp;
        } else if (m_budgetStr == "seconds") {
            m_budgetType = ESeconds;
        } else {
            Assert(false);
        }

        m_budget = props.getFloat("budget", 300.0f);
        m_dumpSDTree = props.getBoolean("dumpSDTree", false);

        m_bsdfSamplingFraction = 0.1f;
        m_productSamplingFraction = 1.0f;
        m_useRR = true;
        m_maxProductAwareBounces = -1;
    }

    ref<BlockedRenderProcess> renderPass(Scene *scene,
        RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        /* This is a sampling-based integrator - parallelize */
        ref<BlockedRenderProcess> proc = new BlockedRenderProcess(job,
            queue, scene->getBlockSize());

        proc->disableProgress();

        proc->bindResource("integrator", integratorResID);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        scene->bindUsedResources(proc);
        bindUsedResources(proc);

        return proc;
    }

    void resetSDTree() {
        Log(EInfo, "Resetting distributions for sampling.");

        m_sdTree->refine((size_t)(std::sqrt(std::pow(2, m_iter) * m_sppPerPass / 4) * m_sTreeThreshold), m_sdTreeMaxMemory);
        m_sdTree->forEachDTreeWrapperParallel([this](DTreeWrapper* dTree) { dTree->reset(20, m_dTreeThreshold); });
    }

    void buildSDTree() {
        Log(EInfo, "Building distributions for sampling.");

        // Build distributions
        m_sdTree->forEachDTreeWrapperParallel([](DTreeWrapper* dTree) { dTree->build(); });

        // Gather statistics
        int maxDepth = 0;
        int minDepth = std::numeric_limits<int>::max();
        Float avgDepth = 0;
        Float maxAvgRadiance = 0;
        Float minAvgRadiance = std::numeric_limits<Float>::max();
        Float avgAvgRadiance = 0;
        size_t maxNodes = 0;
        size_t minNodes = std::numeric_limits<size_t>::max();
        Float avgNodes = 0;
        Float maxStatisticalWeight = 0;
        Float minStatisticalWeight = std::numeric_limits<Float>::max();
        Float avgStatisticalWeight = 0;

        int nPoints = 0;
        int nPointsNodes = 0;

        m_sdTree->forEachDTreeWrapperConst([&](const DTreeWrapper* dTree) {
            const int depth = dTree->depth();
            maxDepth = std::max(maxDepth, depth);
            minDepth = std::min(minDepth, depth);
            avgDepth += depth;

            const Float avgRadiance = dTree->meanRadiance();
            maxAvgRadiance = std::max(maxAvgRadiance, avgRadiance);
            minAvgRadiance = std::min(minAvgRadiance, avgRadiance);
            avgAvgRadiance += avgRadiance;

            if (dTree->numNodes() > 1) {
                const size_t nodes = dTree->numNodes();
                maxNodes = std::max(maxNodes, nodes);
                minNodes = std::min(minNodes, nodes);
                avgNodes += nodes;
                ++nPointsNodes;
            }

            const Float statisticalWeight = dTree->statisticalWeight();
            maxStatisticalWeight = std::max(maxStatisticalWeight, statisticalWeight);
            minStatisticalWeight = std::min(minStatisticalWeight, statisticalWeight);
            avgStatisticalWeight += statisticalWeight;

            ++nPoints;
        });

        if (nPoints > 0) {
            avgDepth /= nPoints;
            avgAvgRadiance /= nPoints;

            if (nPointsNodes > 0) {
                avgNodes /= nPointsNodes;
            }

            avgStatisticalWeight /= nPoints;
        }

        Log(EInfo,
            "Distribution statistics:\n"
            "  Depth         = [%d, %f, %d]\n"
            "  Mean radiance = [%f, %f, %f]\n"
            "  Node count    = [" SIZE_T_FMT ", %f, " SIZE_T_FMT "]\n"
            "  Stat. weight  = [%f, %f, %f]\n",
            minDepth, avgDepth, maxDepth,
            minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
            minNodes, avgNodes, maxNodes,
            minStatisticalWeight, avgStatisticalWeight, maxStatisticalWeight
        );

        m_isBuilt = true;
    }

    void dumpSDTree(Scene* scene, ref<Sensor> sensor) {
        std::ostringstream extension;
        extension << "-" << std::setfill('0') << std::setw(2) << m_iter << ".sdt";
        fs::path path = scene->getDestinationFile();
        path = path.parent_path() / (path.leaf().string() + extension.str());

        auto cameraMatrix = sensor->getWorldTransform()->eval(0).getMatrix();

        BlobWriter blob(path.string());

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                blob << (float)cameraMatrix(i, j);
            }
        }

        m_sdTree->dump(blob);
    }

    bool performRenderPasses(Float& variance, int numPasses, Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        m_image->clear();
        m_squaredImage->clear();

        size_t totalBlocks = 0;

        Log(EInfo, "Rendering %d render passes.", numPasses);

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < numPasses; ++i) {
            ref<BlockedRenderProcess> process = renderPass(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            m_renderProcesses.push_back(process);
            totalBlocks += process->totalBlocks();
        }

        bool result = true;
        int passesRenderedLocal = 0;

        static const size_t processBatchSize = 128;

        for (size_t i = 0; i < m_renderProcesses.size(); i += processBatchSize) {
            const size_t start = i;
            const size_t end = std::min(i + processBatchSize, m_renderProcesses.size());
            for (size_t j = start; j < end; ++j) {
                sched->schedule(m_renderProcesses[j]);
            }

            for (size_t j = start; j < end; ++j) {
                auto& process = m_renderProcesses[j];
                sched->wait(process);

                ++m_passesRendered;
                ++m_passesRenderedThisIter;
                ++passesRenderedLocal;

                int progress = 0;
                bool shouldAbort;
                switch (m_budgetType) {
                    case ESpp:
                        progress = m_passesRendered;
                        shouldAbort = false;
                        break;
                    case ESeconds:
                        progress = (int)computeElapsedSeconds(m_startTime);
                        shouldAbort = progress > m_budget;
                        break;
                    default:
                        Assert(false);
                        break;
                }

                m_progress->update(progress);

                if (process->getReturnStatus() != ParallelProcess::ESuccess) {
                    result = false;
                    shouldAbort = true;
                }

                if (shouldAbort) {
                    goto l_abort;
                }
            }
        }
    l_abort:

        for (auto& process : m_renderProcesses) {
            sched->cancel(process);
        }

        m_renderProcesses.clear();

        variance = 0;
        Bitmap* squaredImage = m_squaredImage->getBitmap();
        Bitmap* image = m_image->getBitmap();

        if (m_sampleCombination == ESampleCombination::EInverseVariance) {
            // Record all previously rendered iterations such that later on all iterations can be
            // combined by weighting them by their estimated inverse pixel variance.
            m_images.push_back(image->clone());
        }

        m_varianceBuffer->clear();

        int N = passesRenderedLocal * m_sppPerPass;

        Vector2i size = squaredImage->getSize();
        for (int x = 0; x < size.x; ++x)
            for (int y = 0; y < size.y; ++y) {
                Point2i pos = Point2i(x, y);
                Spectrum pixel = image->getPixel(pos);
                Spectrum localVar = squaredImage->getPixel(pos) - pixel * pixel / (Float)N;
                image->setPixel(pos, localVar);
                // The local variance is clamped such that fireflies don't cause crazily unstable estimates.
                variance += std::min(localVar.getLuminance(), 10000.0f);
            }

        variance /= (Float)size.x * size.y * (N - 1);
        m_varianceBuffer->put(m_image);

        if (m_sampleCombination == ESampleCombination::EInverseVariance) {
            // Record estimated mean pixel variance for later use in weighting of all images.
            m_variances.push_back(variance);
        }

        Float seconds = computeElapsedSeconds(start);

        const Float ttuv = seconds * variance;
        const Float stuv = passesRenderedLocal * m_sppPerPass * variance;
        Log(EInfo, "%.2f seconds, Total passes: %d, Var: %f, TTUV: %f, STUV: %f.",
            seconds, m_passesRendered, variance, ttuv, stuv);

        return result;
    }

    bool doNeeWithSpp(int spp) {
        switch (m_nee) {
            case ENever:
                return false;
            case EKickstart:
                return spp < 128;
            default:
                return true;
        }
    }

    bool renderSPP(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t sampleCount = (size_t)m_budget;

        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        int nPasses = (int)std::ceil(sampleCount / (Float)m_sppPerPass);
        sampleCount = m_sppPerPass * nPasses;

        bool result = true;
        Float currentVarAtEnd = std::numeric_limits<Float>::infinity();

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", nPasses, job));

        while (result && m_passesRendered < nPasses) {
            const int sppRendered = m_passesRendered * m_sppPerPass;
            m_doNee = doNeeWithSpp(sppRendered);

            int remainingPasses = nPasses - m_passesRendered;
            int passesThisIteration = std::min(remainingPasses, 1 << m_iter);

            // If the next iteration does not manage to double the number of passes once more
            // then it would be unwise to throw away the current iteration. Instead, extend
            // the current iteration to the end.
            // This condition can also be interpreted as: the last iteration must always use
            // at _least_ half the total sample budget.
            if (remainingPasses - passesThisIteration < 2 * passesThisIteration) {
                passesThisIteration = remainingPasses;
            }

            Log(EInfo, "ITERATION %d, %d passes", m_iter, passesThisIteration);
            
            m_isFinalIter = passesThisIteration >= remainingPasses;

            film->clear();
            resetSDTree();

            Float variance;
            if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                result = false;
                break;
            }

            const Float lastVarAtEnd = currentVarAtEnd;
            currentVarAtEnd = passesThisIteration * variance / remainingPasses;

            Log(EInfo,
                "Extrapolated var:\n"
                "  Last:    %f\n"
                "  Current: %f\n",
                lastVarAtEnd, currentVarAtEnd);

            remainingPasses -= passesThisIteration;
            if (m_sampleCombination == ESampleCombination::EDiscardWithAutomaticBudget && remainingPasses > 0 && (
                    // if there is any time remaining we want to keep going if
                    // either will have less time next iter
                    remainingPasses < passesThisIteration ||
                    // or, according to the convergence behavior, we're better off if we keep going
                    // (we only trust the variance if we drew enough samples for it to be a reliable estimate,
                    // captured by an arbitraty threshold).
                    (sppRendered > 256 && currentVarAtEnd > lastVarAtEnd)
                )) {
                Log(EInfo, "FINAL %d passes", remainingPasses);
                m_isFinalIter = true;
                if (!performRenderPasses(variance, remainingPasses, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                    result = false;
                    break;
                }
            }
            buildSDTree();

            if (m_dumpSDTree) {
                dumpSDTree(scene, sensor);
            }

            ++m_iter;
            m_passesRenderedThisIter = 0;
        }

        return result;
    }

    static Float computeElapsedSeconds(std::chrono::steady_clock::time_point start) {
        auto current = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
        return (Float)ms.count() / 1000;
    }

    bool renderTime(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        Float nSeconds = m_budget;

        bool result = true;
        Float currentVarAtEnd = std::numeric_limits<Float>::infinity();

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", (int)nSeconds, job));

        Float elapsedSeconds = 0;

        while (result && elapsedSeconds < nSeconds) {
            const int sppRendered = m_passesRendered * m_sppPerPass;
            m_doNee = doNeeWithSpp(sppRendered);

            Float remainingTime = nSeconds - elapsedSeconds;
            const int passesThisIteration = 1 << m_iter;

            Log(EInfo, "ITERATION %d, %d passes", m_iter, passesThisIteration);

            const auto startIter = std::chrono::steady_clock::now();

            film->clear();
            resetSDTree();

            Float variance;
            if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                result = false;
                break;
            }

            const Float secondsIter = computeElapsedSeconds(startIter);

            const Float lastVarAtEnd = currentVarAtEnd;
            currentVarAtEnd = secondsIter * variance / remainingTime;

            Log(EInfo,
                "Extrapolated var:\n"
                "  Last:    %f\n"
                "  Current: %f\n",
                lastVarAtEnd, currentVarAtEnd);

            remainingTime -= secondsIter;
            if (m_sampleCombination == ESampleCombination::EDiscardWithAutomaticBudget && remainingTime > 0 && (
                    // if there is any time remaining we want to keep going if
                    // either will have less time next iter
                    remainingTime < secondsIter ||
                    // or, according to the convergence behavior, we're better off if we keep going
                    // (we only trust the variance if we drew enough samples for it to be a reliable estimate,
                    // captured by an arbitraty threshold).
                    (sppRendered > 256 && currentVarAtEnd > lastVarAtEnd)
                )) {
                Log(EInfo, "FINAL %f seconds", remainingTime);
                m_isFinalIter = true;
                do {
                    if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                        result = false;
                        break;
                    }

                    elapsedSeconds = computeElapsedSeconds(m_startTime);
                } while (elapsedSeconds < nSeconds);
            }
            buildSDTree();

            if (m_dumpSDTree) {
                dumpSDTree(scene, sensor);
            }

            ++m_iter;
            m_passesRenderedThisIter = 0;
            elapsedSeconds = computeElapsedSeconds(m_startTime);
        }

        return result;
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) {

        m_sdTree = std::unique_ptr<STree>(new STree(scene->getAABB()));
        m_iter = 0;
        m_isFinalIter = false;

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t nCores = sched->getCoreCount();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        auto properties = Properties("hdrfilm");
        properties.setInteger("width", film->getSize().x);
        properties.setInteger("height", film->getSize().y);
        m_varianceBuffer = static_cast<Film*>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
        m_varianceBuffer->setDestinationFile(scene->getDestinationFile(), 0);

        m_squaredImage = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());
        m_image = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());

        m_images.clear();
        m_variances.clear();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y, nCores, nCores == 1 ? "core" : "cores");

        Thread::initializeOpenMP(nCores);

        int integratorResID = sched->registerResource(this);
        bool result = true;

        m_startTime = std::chrono::steady_clock::now();

        m_passesRendered = 0;
        switch (m_budgetType) {
            case ESpp:
                result = renderSPP(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
                break;
            case ESeconds:
                result = renderTime(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
                break;
            default:
                Assert(false);
                break;
        }

        sched->unregisterResource(integratorResID);

        m_progress = nullptr;

        if (m_sampleCombination == ESampleCombination::EInverseVariance) {
            // Combine the last 4 images according to their inverse variance
            film->clear();
            ref<ImageBlock> tmp = new ImageBlock(Bitmap::ESpectrum, film->getCropSize());
            size_t begin = m_images.size() - std::min(m_images.size(), (size_t)4);

            Float totalWeight = 0;
            for (size_t i = begin; i < m_variances.size(); ++i) {
                totalWeight += 1.0f / m_variances[i];
            }

            for (size_t i = begin; i < m_images.size(); ++i) {
                m_images[i]->convert(tmp->getBitmap(), 1.0f / m_variances[i] / totalWeight);
                film->addBitmap(tmp->getBitmap());
            }
        }

        return result;
    }

    void renderBlock(const Scene *scene, const Sensor *sensor,
        Sampler *sampler, ImageBlock *block, const bool &stop,
        const std::vector< TPoint2<uint8_t> > &points) const {

        Float diffScaleFactor = 1.0f /
            std::sqrt((Float)m_sppPerPass);

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

        ref<ImageBlock> squaredBlock = new ImageBlock(block->getPixelFormat(), block->getSize(), block->getReconstructionFilter());
        squaredBlock->setOffset(block->getOffset());
        squaredBlock->clear();

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) // Don't compute an alpha channel if we don't have to
            queryType &= ~RadianceQueryRecord::EOpacity;

        for (size_t i = 0; i < points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            if (stop)
                break;

            for (int j = 0; j < m_sppPerPass; j++) {
                rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                Spectrum spec = sensor->sampleRayDifferential(
                    sensorRay, samplePos, apertureSample, timeSample);

                sensorRay.scaleDifferential(diffScaleFactor);

                spec *= Li(sensorRay, rRec);
                block->put(samplePos, spec, rRec.alpha);
                squaredBlock->put(samplePos, spec * spec, rRec.alpha);
                sampler->advance();
            }
        }

        m_squaredImage->put(squaredBlock);
        m_image->put(block);
    }

    void cancel() {
        const auto& scheduler = Scheduler::getInstance();
        for (size_t i = 0; i < m_renderProcesses.size(); ++i) {
            scheduler->cancel(m_renderProcesses[i]);
        }
    }

    void setGuidingMode(const BSDF *bsdf, const BSDFSamplingRecord &bRec, RadianceProxy &radianceProxy, const DTreeWrapper *dTree, Float &bsdfSamplingFraction, Float &productSamplingFraction, EGuidingMode &guidingMode, EGuidingMode& bounceMode, int& maxProductAwareBounces) const
    {
        auto type = bsdf->getType();
        const bool canUseGuiding = m_isBuilt && dTree && (type & BSDF::EDelta) != (type & BSDF::EAll);

        if (!canUseGuiding)
        {
            bsdfSamplingFraction = 1.0f;
            productSamplingFraction = 0.0f;
            bounceMode = EGuidingMode::ECombined;
        }
        else
        {
            // Fixed fraction case
            if (m_bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::ENone)
            {

                if (m_guidingMode == EGuidingMode::EProduct)
                {
                    // Try Init product guiding
                    BSDFProxy bsdfProxy;
                    bool flipNormal;
                    const bool useProductGuiding = radianceProxy.is_built() && bsdf->add_parameters_to_proxy(bsdfProxy, bRec, flipNormal);
                    const Vector proxyNormal = flipNormal ? -Vector(bRec.its.shFrame.n) : Vector(bRec.its.shFrame.n);

                    if (useProductGuiding)
                    {
                        radianceProxy.build_product(bsdfProxy, bRec.its.toWorld(bRec.wi), proxyNormal);
                        bsdfSamplingFraction = m_bsdfSamplingFraction;
                        productSamplingFraction = 1.0f;
                    }
                    else
                    {
                        bsdfSamplingFraction = 1.0f;
                        productSamplingFraction = 0.0f;
                    }
                }
                else
                {
                    bsdfSamplingFraction = m_bsdfSamplingFraction;
                    productSamplingFraction = 0.0f;
                }
            }
            // Learned fraction
            else
            {
                if (maxProductAwareBounces != 0 && (m_guidingMode == EGuidingMode::EProduct || m_guidingMode == EGuidingMode::ECombined))
                {
                    // Try Init product guiding
                    BSDFProxy bsdfProxy;
                    bool flipNormal;
                    const bool useProductGuiding = radianceProxy.is_built() && bsdf->add_parameters_to_proxy(bsdfProxy, bRec, flipNormal);
                    const Vector proxyNormal = flipNormal ? -Vector(bRec.its.shFrame.n) : Vector(bRec.its.shFrame.n);

                    if (useProductGuiding)
                    {
                        radianceProxy.build_product(bsdfProxy, bRec.its.toWorld(bRec.wi), proxyNormal);
                        --maxProductAwareBounces;
                    }
                    else // Path guiding
                    {
                        bounceMode = guidingMode = EGuidingMode::ERadiance;
                        bsdfSamplingFraction = dTree->bsdfSamplingFractionRadiance();
                        productSamplingFraction = 0.0f;
                        return;
                    }

                    if (m_guidingMode == EGuidingMode::EProduct)
                    {
                        bounceMode = guidingMode = EGuidingMode::EProduct;
                        bsdfSamplingFraction = useProductGuiding ? dTree->bsdfSamplingFractionProduct() : 1.0f;
                        productSamplingFraction = useProductGuiding ? 1.0f : 0.0f;
                    }
                    else
                    {
                        if (useProductGuiding)
                        {
                            guidingMode = EGuidingMode::ECombined;
                            const Vector2 samplingFractions = dTree->bsdfSamplingFractionCombined();
                            bsdfSamplingFraction = samplingFractions.x;
                            productSamplingFraction = samplingFractions.y;
                        }
                        else
                        {
                            guidingMode = EGuidingMode::ERadiance;
                            bsdfSamplingFraction = dTree->bsdfSamplingFractionRadiance();
                            productSamplingFraction = 0.0f;
                        }
                    }
                }
                else // Path guiding
                {
                    bounceMode = guidingMode = EGuidingMode::ERadiance;
                    bsdfSamplingFraction = dTree->bsdfSamplingFractionRadiance();
                    productSamplingFraction = 0.0f;
                }
            }
        }
    }

    Spectrum sampleMat(const BSDF* bsdf, const RadianceProxy& radianceProxy, BSDFSamplingRecord& bRec, Float& woPdf, Float& bsdfPdf, Float& dTreePdf, Float& productPdf, const Float bsdfSamplingFraction, const Float productSamplingFraction, RadianceQueryRecord& rRec, const DTreeWrapper* dTree) const {
        Point2 sample = rRec.nextSample2D();

        // auto type = bsdf->getType();
        // if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll)) {
        //     auto result = bsdf->sample(bRec, bsdfPdf, sample);
        //     woPdf = bsdfPdf;
        //     productPdf = dTreePdf = 0;
        //     return result;
        // }

        Spectrum result;
        if (sample.x < bsdfSamplingFraction) {
            sample.x /= bsdfSamplingFraction;
            result = bsdf->sample(bRec, bsdfPdf, sample);
            if (result.isZero()) {
                woPdf = bsdfPdf = dTreePdf = productPdf = 0;
                return Spectrum{0.0f};
            }

            // If we sampled a delta component, then we have a 0 probability
            // of sampling that direction via guiding, thus we can return early.
            if (bRec.sampledType & BSDF::EDelta) {
                productPdf = dTreePdf = 0;
                woPdf = bsdfPdf * bsdfSamplingFraction;
                return result / bsdfSamplingFraction;
            }

            result *= bsdfPdf;
        } else {
            // Product guiding
            if (sample.y < productSamplingFraction)
            {
                Vector wo;
                radianceProxy.sample(rRec.sampler, wo);
                bRec.wo = bRec.its.toLocal(wo);
            }
            // else D-Tree guiding
            else
            {
                bRec.wo = bRec.its.toLocal(dTree->sample(rRec.sampler));
            }
            result = bsdf->eval(bRec);
        }

        pdfMat(woPdf, bsdfPdf, dTreePdf, productPdf, bsdfSamplingFraction, productSamplingFraction, bsdf, radianceProxy, bRec, dTree);
        if (woPdf == 0) {
            return Spectrum{0.0f};
        }

        return result / woPdf;
    }

    void pdfMat(Float &woPdf, Float &bsdfPdf, Float &dTreePdf, Float &productPdf, Float bsdfSamplingFraction, Float productSamplingFraction, const BSDF *bsdf, const RadianceProxy &radianceProxy, const BSDFSamplingRecord &bRec, const DTreeWrapper *dTree) const
    {
        productPdf = dTreePdf = 0;

        auto type = bsdf->getType();
        if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll)) {
            woPdf = bsdfPdf = bsdf->pdf(bRec);
            productPdf = dTreePdf = 0;
            return;
        }

        bsdfPdf = bsdf->pdf(bRec);
        if (!std::isfinite(bsdfPdf)) {
            woPdf = bsdfPdf = 0;
            return;
        }

        dTreePdf = productSamplingFraction != 1.0f ? dTree->pdf(bRec.its.toWorld(bRec.wo)) : 0.0f;
        productPdf = productSamplingFraction == 0.0f ? 0.0f : radianceProxy.pdf(bRec.its.toWorld(bRec.wo));
            
        woPdf = bsdfSamplingFraction * bsdfPdf + (1.0f - bsdfSamplingFraction) * (productSamplingFraction * productPdf + (1.0f - productSamplingFraction) * dTreePdf);
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        struct Vertex {
            DTreeWrapper* dTree;
            Vector dTreeVoxelSize;
            Ray ray;

            Spectrum throughput;
            Spectrum bsdfVal;

            Spectrum radiance;

            Float woPdf, bsdfPdf, dTreePdf, productPdf;
            EGuidingMode guidingMode;
            bool isDelta;
            EGuidingMode bounceMode;

            void record(const Spectrum& r) {
                radiance += r;
            }

            void recordMeasurement(Spectrum &r) {
                dTree->recordMeasurement(r);
            }

            void commit(STree& sdTree, Float statisticalWeight, ESpatialFilter spatialFilter, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler* sampler) {
                if (!(woPdf > 0) || !radiance.isValid() || !bsdfVal.isValid()) {
                    return;
                }

                Spectrum localRadiance = Spectrum{0.0f};
                if (throughput[0] * woPdf > Epsilon) localRadiance[0] = radiance[0] / throughput[0];
                if (throughput[1] * woPdf > Epsilon) localRadiance[1] = radiance[1] / throughput[1];
                if (throughput[2] * woPdf > Epsilon) localRadiance[2] = radiance[2] / throughput[2];
                Spectrum product = localRadiance * bsdfVal;

                DTreeRecord rec{ray.d, localRadiance.average(), product.average(), woPdf, bsdfPdf, dTreePdf, productPdf, guidingMode, statisticalWeight, isDelta, bounceMode};
                switch (spatialFilter) {
                    case ESpatialFilter::ENearest:
                        dTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                        break;
                    case ESpatialFilter::EStochasticBox:
                        {
                            DTreeWrapper* splatDTree = dTree;

                            // Jitter the actual position within the
                            // filter box to perform stochastic filtering.
                            Vector offset = dTreeVoxelSize;
                            offset.x *= sampler->next1D() - 0.5f;
                            offset.y *= sampler->next1D() - 0.5f;
                            offset.z *= sampler->next1D() - 0.5f;

                            Point origin = sdTree.aabb().clip(ray.o + offset);
                            splatDTree = sdTree.dTreeWrapper(origin);
                            if (splatDTree) {
                                splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                            }
                            break;
                        }
                    case ESpatialFilter::EBox:
                        sdTree.record(ray.o, dTreeVoxelSize, rec, directionalFilter, bsdfSamplingFractionLoss);
                        break;
                }
            }
        };

        static const int MAX_NUM_VERTICES = 32;
        std::array<Vertex, MAX_NUM_VERTICES> vertices;

        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        MediumSamplingRecord mRec;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Float eta = 1.0f;

        /* Perform the first ray intersection (or ignore if the
        intersection has already been provided). */
        rRec.rayIntersect(ray);

        Spectrum throughput(1.0f);
        Spectrum measurementEstimate;
        bool scattered = false;

        int nVertices = 0;

        auto recordRadiance = [&](Spectrum radiance) {
            //if (!(m_isFinalIter && rRec.depth <= 1))
                Li += radiance;

            for (int i = 0; i < nVertices; ++i)
            {
                vertices[i].record(radiance);
            }
        };

        // Only render indirect
        // rRec.type = RadianceQueryRecord::ESubsurfaceRadiance;

        int maxProductAwareBounces = m_maxProductAwareBounces;

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0)
        {

            /* ==================================================================== */
            /*                 Radiative Transfer Equation sampling                 */
            /* ==================================================================== */
            if (rRec.medium && rRec.medium->sampleDistance(Ray(ray, 0, its.t), mRec, rRec.sampler)) {
                /* Sample the integral
                \int_x^y tau(x, x') [ \sigma_s \int_{S^2} \rho(\omega,\omega') L(x,\omega') d\omega' ] dx'
                */
                const PhaseFunction *phase = mRec.getPhaseFunction();

                if (rRec.depth >= m_maxDepth && m_maxDepth != -1) // No more scattering events allowed
                    break;

                throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

                /* ==================================================================== */
                /*                          Luminaire sampling                          */
                /* ==================================================================== */

                /* Estimate the single scattering component if this is requested */
                DirectSamplingRecord dRec(mRec.p, mRec.time);

                if (rRec.type & RadianceQueryRecord::EDirectMediumRadiance) {
                    int interactions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                        dRec, rRec.medium, interactions,
                        rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero()) {
                        const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                        /* Evaluate the phase function */
                        PhaseFunctionSamplingRecord pRec(mRec, -ray.d, dRec.d);
                        Float phaseVal = phase->eval(pRec);

                        if (phaseVal != 0) {
                            /* Calculate prob. of having sampled that direction using
                            phase function sampling */
                            Float phasePdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                                ? phase->pdf(pRec) : (Float) 0.0f;

                            /* Weight using the power heuristic */
                            const Float weight = miWeight(dRec.pdf, phasePdf);
                            recordRadiance(throughput * value * phaseVal * weight);
                        }
                    }
                }

                /* ==================================================================== */
                /*                         Phase function sampling                      */
                /* ==================================================================== */

                Float phasePdf;
                PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
                Float phaseVal = phase->sample(pRec, phasePdf, rRec.sampler);
                if (phaseVal == 0)
                    break;
                throughput *= phaseVal;

                /* Trace a ray in this direction */
                ray = Ray(mRec.p, pRec.wo, ray.time);
                ray.mint = 0;

                Spectrum value(0.0f);
                rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                    m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

                /* If a luminaire was hit, estimate the local illumination and
                weight using the power heuristic */
                if (!value.isZero() && (rRec.type & RadianceQueryRecord::EDirectMediumRadiance)) {
                    const Float emitterPdf = scene->pdfEmitterDirect(dRec);
                    recordRadiance(throughput * value * miWeight(phasePdf, emitterPdf));
                }

                /* ==================================================================== */
                /*                         Multiple scattering                          */
                /* ==================================================================== */

                /* Stop if multiple scattering was not requested */
                if (!(rRec.type & RadianceQueryRecord::EIndirectMediumRadiance))
                    break;
                rRec.type = RadianceQueryRecord::ERadianceNoEmission;

                if (rRec.depth++ >= m_rrDepth) {
                    /* Russian roulette: try to keep path weights equal to one,
                    while accounting for the solid angle compression at refractive
                    index boundaries. Stop with at least some probability to avoid
                    getting stuck (e.g. due to total internal reflection) */

                    Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                    if (rRec.nextSample1D() >= q)
                        break;
                    throughput /= q;
                }
            } else {
                /* Sample
                tau(x, y) (Surface integral). This happens with probability mRec.pdfFailure
                Account for this and multiply by the proper per-color-channel transmittance.
                */
                if (rRec.medium)
                    throughput *= mRec.transmittance / mRec.pdfFailure;

                if (!its.isValid()) {
                    /* If no intersection could be found, possibly return
                    attenuated radiance from a background luminaire */
                    if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                        && (!m_hideEmitters || scattered)) {
                        Spectrum value = throughput * scene->evalEnvironment(ray);
                        if (rRec.medium)
                            value *= rRec.medium->evalTransmittance(ray, rRec.sampler);
                        recordRadiance(value);
                    }

                    break;
                }

                /* Possibly include emitted radiance if requested */
                if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    recordRadiance(throughput * its.Le(-ray.d));

                /* Include radiance from a subsurface integrator if requested */
                if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                    recordRadiance(throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth));

                if (rRec.depth >= m_maxDepth && m_maxDepth != -1)
                    break;

                /* Prevent light leaks due to the use of shading normals */
                Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
                    wiDotShN = Frame::cosTheta(its.wi);
                if (wiDotGeoN * wiDotShN < 0 && m_strictNormals)
                    break;

                const BSDF *bsdf = its.getBSDF();

                Vector dTreeVoxelSize;
                DTreeWrapper* dTree = nullptr;

                // We only guide smooth BRDFs for now. Analytic product sampling
                // would be conceivable for discrete decisions such as refraction vs
                // reflection.
                if (bsdf->getType() & BSDF::ESmooth) {
                    dTree = m_sdTree->dTreeWrapper(its.p, dTreeVoxelSize);
                }

                Float bsdfSamplingFraction = m_bsdfSamplingFraction;
                Float productSamplingFraction;
                // if (dTree && m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone) {
                //     bsdfSamplingFraction = dTree->bsdfSamplingFraction();
                // }

                /* ==================================================================== */
                /*                            BSDF sampling                             */
                /* ==================================================================== */

                /* Sample BSDF * cos(theta) */
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                Float woPdf, bsdfPdf, dTreePdf, productPdf;
                EGuidingMode guidingMode;
                EGuidingMode bounceMode;
                RadianceProxy radianceProxy;
                if (dTree)
                    radianceProxy.set_maps(dTree->getRadianceProxy());

                setGuidingMode(bsdf, bRec, radianceProxy, dTree, bsdfSamplingFraction, productSamplingFraction, guidingMode, bounceMode, maxProductAwareBounces);

                Spectrum bsdfWeight = sampleMat(bsdf, radianceProxy, bRec, woPdf, bsdfPdf, dTreePdf, productPdf, bsdfSamplingFraction, productSamplingFraction, rRec, dTree);

                // Visualize RadianceProxy
                // if (m_isFinalIter && dTree)
                // {
                //     RadianceProxy radianceProxy(dTree->getRadianceProxy());
                //     if (radianceProxy.is_built())
                //     {
                //         const Vector n = its.shFrame.n;
                //         const Vector dir = bRec.its.toWorld(bRec.wo);
                //         const float radiance = radianceProxy.proxy_radiance(dir);
                //         // const float radiance = dTree->radiance(dir);
                //         BSDFProxy bsdfProxy;
                //         bool flipNormal;
                //         const bool useBsdfProxy = bsdf->add_parameters_to_proxy(bsdfProxy, bRec, flipNormal);
                //         Spectrum Li;
                //         if (useBsdfProxy)
                //         {

                //             // bsdfProxy.finish_parameterization(bRec.its.toWorld(bRec.wi), flipNormal ? -n : n);
                //             // const float bsdfVal = woPdf == 0 ? 0.0f : bsdfProxy.evaluate(dir) / woPdf;
                //             // Li = (radiance * bsdfVal) * throughput;

                //             radianceProxy.build_product(bsdfProxy, bRec.its.toWorld(bRec.wi), flipNormal ? -n : n);
                //             Vector wo;
                //             const float productPdf = radianceProxy.sample(rRec.sampler, wo);
                //             const float product = productPdf == 0 ? 0.0f : radianceProxy.proxy_radiance(wo) / productPdf;
                //             Li = product * throughput;
                //         }
                //         else
                //         {
                //             // Li = radiance * bsdfWeight * throughput;
                //             // Li = radiance * bsdfWeight.average() * throughput;
                //             Spectrum Li;
                //             Li.fromLinearRGB(0.0f, 1.0f, 1.0f);
                //             return Li;
                //         }

                //         return Li;
                //     }
                //     else
                //     {
                //         Spectrum Li;
                //         Li.fromLinearRGB(0.0f, 0.0f, 1.0f);
                //         return Li;
                //     }
                // }
                // else if (m_isFinalIter && !dTree)
                // {
                //     Spectrum Li;
                //     Li.fromLinearRGB(1.0f, 1.0f, 0.0f);
                //     return Li;
                // }

                /* ==================================================================== */
                /*                          Luminaire sampling                          */
                /* ==================================================================== */

                DirectSamplingRecord dRec(its);

                /* Estimate the direct illumination if this is requested */
                if (m_doNee &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                    (bsdf->getType() & BSDF::ESmooth)) {
                    int interactions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                        dRec, its, rRec.medium, interactions,
                        rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero()) {
                        BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));

                        Float woDotGeoN = dot(its.geoFrame.n, dRec.d);

                        /* Prevent light leaks due to the use of shading normals */
                        if (!m_strictNormals || woDotGeoN * Frame::cosTheta(bRec.wo) > 0) {
                            /* Evaluate BSDF * cos(theta) */
                            const Spectrum bsdfVal = bsdf->eval(bRec);

                            /* Calculate prob. of having generated that direction using BSDF sampling */
                            const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                            Float woPdf = 0, bsdfPdf = 0, dTreePdf = 0;
                            if (emitter->isOnSurface() && dRec.measure == ESolidAngle) {
                                pdfMat(woPdf, bsdfPdf, dTreePdf, productPdf, bsdfSamplingFraction, productSamplingFraction, bsdf, radianceProxy, bRec, dTree);
                            }

                            /* Weight using the power heuristic */
                            const Float weight = miWeight(dRec.pdf, woPdf);

                            value *= bsdfVal;
                            Spectrum L = throughput * value * weight;

                            if (!m_isFinalIter && m_nee != EAlways) {
                                if (dTree) {
                                    Vertex v = Vertex{
                                        dTree,
                                        dTreeVoxelSize,
                                        Ray(its.p, dRec.d, 0),
                                        throughput * bsdfVal / dRec.pdf,
                                        bsdfVal,
                                        L,
                                        dRec.pdf,
                                        bsdfPdf,
                                        dTreePdf,
                                        productPdf,
                                        guidingMode,
                                        false,
                                        bounceMode
                                    };

                                    v.commit(*m_sdTree, 0.5f, m_spatialFilter, m_directionalFilter, m_isBuilt ? m_bsdfSamplingFractionLoss : EBsdfSamplingFractionLoss::ENone, rRec.sampler);
                                }
                            }
                            
                            recordRadiance(L);
                        }
                    }
                }

                // BSDF handling
                if (bsdfWeight.isZero())
                    break;

                /* Prevent light leaks due to the use of shading normals */
                const Vector wo = its.toWorld(bRec.wo);
                Float woDotGeoN = dot(its.geoFrame.n, wo);

                if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_strictNormals)
                    break;

                /* Trace a ray in this direction */
                ray = Ray(its.p, wo, ray.time);

                /* Keep track of the throughput, medium, and relative
                refractive index along the path */
                throughput *= bsdfWeight;
                eta *= bRec.eta;
                if (its.isMediumTransition())
                    rRec.medium = its.getTargetMedium(ray.d);

                /* Handle index-matched medium transitions specially */
                if (bRec.sampledType == BSDF::ENull) {
                    if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                        break;

                    // There exist materials that are smooth/null hybrids (e.g. the mask BSDF), which means that
                    // for optimal-sampling-fraction optimization we need to record null transitions for such BSDFs.
                    if (m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone && dTree && nVertices < MAX_NUM_VERTICES && !m_isFinalIter) {
                        if (1 / woPdf > 0) {
                            vertices[nVertices] = Vertex{
                                dTree,
                                dTreeVoxelSize,
                                ray,
                                throughput,
                                bsdfWeight * woPdf,
                                Spectrum{0.0f},
                                woPdf,
                                bsdfPdf,
                                dTreePdf,
                                productPdf,
                                guidingMode,
                                true,
                                EGuidingMode::ECombined
                            };

                            ++nVertices;
                        }
                    }

                    rRec.type = scattered ? RadianceQueryRecord::ERadianceNoEmission
                        : RadianceQueryRecord::ERadiance;
                    scene->rayIntersect(ray, its);
                    rRec.depth++;
                    continue;
                }

                Spectrum value(0.0f);
                rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                    m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

                /* If a luminaire was hit, estimate the local illumination and
                weight using the power heuristic */
                if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) {
                    bool isDelta = bRec.sampledType & BSDF::EDelta;
                    const Float emitterPdf = (m_doNee && !isDelta && !value.isZero()) ? scene->pdfEmitterDirect(dRec) : 0;

                    const Float weight = miWeight(woPdf, emitterPdf);
                    Spectrum L = throughput * value * weight;
                    if (!L.isZero()) {
                        recordRadiance(L);
                    }

                    if ((!isDelta || m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone) && dTree && nVertices < MAX_NUM_VERTICES && !m_isFinalIter) {

                        if (nVertices == 0 && m_isBuilt) {
                            measurementEstimate = dTree->measurementEstimate();
                        }

                        if (1 / woPdf > 0) {
                            vertices[nVertices] = Vertex{
                                dTree,
                                dTreeVoxelSize,
                                ray,
                                throughput,
                                bsdfWeight * woPdf,
                                (m_nee == EAlways) ? Spectrum{0.0f} : L,
                                woPdf,
                                bsdfPdf,
                                dTreePdf,
                                productPdf,
                                guidingMode,
                                isDelta,
                                bounceMode
                            };

                            ++nVertices;
                        }
                    }
                }

                /* ==================================================================== */
                /*                         Indirect illumination                        */
                /* ==================================================================== */

                /* Stop if indirect illumination was not requested */
                if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                    break;

                rRec.type = RadianceQueryRecord::ERadianceNoEmission;

                // Russian roulette
                // The adjoint driven Russian Roulette implementation was taken from
                // "Practical Product Path Guiding Using Linearly Transformed Cosines" [Diolatzis et. al. 2020]
                // Implementation at https://gitlab.inria.fr/sdiolatz/practical-product-path-guiding
                if (rRec.depth++ >= m_rrDepth) {
                    Float successProb = 1.0f;
                    if (dTree && !(bRec.sampledType & BSDF::EDelta)) {
                        if (!m_isBuilt) {
                            successProb = throughput.max() * eta * eta;
                        } else {
                            if (m_useRR)
                            {
                                Spectrum incidentRadiance = Spectrum{dTree->estimateRadiance(ray.d)};

                                if (measurementEstimate.min() > 0 && incidentRadiance.min() > 0)
                                {
                                    const Float center = (measurementEstimate / incidentRadiance).average();
                                    const Float s = 5;
                                    const Float wMin = 2 * center / (1 + s);
                                    //const Float wMax = wMin * 5; // Useful for splitting

                                    const Float tMax = throughput.max();
                                    //const Float tMin = throughput.min(); // Useful for splitting

                                    // Splitting is not supported.
                                    if (tMax < wMin)
                                    {
                                        successProb = tMax / wMin;
                                    }
                                }
                            }
                        }

                        successProb = std::max(0.1f, std::min(successProb, 0.99f));
                    } else {
                        // In some case, we can end up to infinite loop when maxDepth = -1
                        // due to very very long specular chain. In this case,
                        // we do RR anyway in the classical way if exceeding very long paths
                        if (rRec.depth > 40) {
                            SLog(EWarn, "Reaching maximum depth for specular chain. Activating RR");
                            successProb = throughput.max() * eta * eta;
                        }
                    }

                    if (rRec.nextSample1D() >= successProb)
                        break;
                    throughput /= successProb;
                }
            }

            scattered = true;
        }
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        if (nVertices > 0 && !m_isFinalIter) {
            vertices[0].recordMeasurement(Li);
            for (int i = 0; i < nVertices; ++i) {
                vertices[i].commit(*m_sdTree, m_nee == EKickstart && m_doNee ? 0.5f : 1.0f, m_spatialFilter, m_directionalFilter, m_isBuilt ? m_bsdfSamplingFractionLoss : EBsdfSamplingFractionLoss::ENone, rRec.sampler);
            }
        }

        return Li;
    }

    /**
    * This function is called by the recursive ray tracing above after
    * having sampled a direction from a BSDF/phase function. Due to the
    * way in which this integrator deals with index-matched boundaries,
    * it is necessarily a bit complicated (though the improved performance
    * easily pays for the extra effort).
    *
    * This function
    *
    * 1. Intersects 'ray' against the scene geometry and returns the
    *    *first* intersection via the '_its' argument.
    *
    * 2. It checks whether the intersected shape was an emitter, or if
    *    the ray intersects nothing and there is an environment emitter.
    *    In this case, it returns the attenuated emittance, as well as
    *    a DirectSamplingRecord that can be used to query the hypothetical
    *    sampling density at the emitter.
    *
    * 3. If current shape is an index-matched medium transition, the
    *    integrator keeps on looking on whether a light source eventually
    *    follows after a potential chain of index-matched medium transitions,
    *    while respecting the specified 'maxDepth' limits. It then returns
    *    the attenuated emittance of this light source, while accounting for
    *    all attenuation that occurs on the wya.
    */
    void rayIntersectAndLookForEmitter(const Scene *scene, Sampler *sampler,
        const Medium *medium, int maxInteractions, Ray ray, Intersection &_its,
        DirectSamplingRecord &dRec, Spectrum &value) const {
        Intersection its2, *its = &_its;
        Spectrum transmittance(1.0f);
        bool surface = false;
        int interactions = 0;

        while (true) {
            surface = scene->rayIntersect(ray, *its);

            if (medium)
                transmittance *= medium->evalTransmittance(Ray(ray, 0, its->t), sampler);

            if (surface && (interactions == maxInteractions ||
                !(its->getBSDF()->getType() & BSDF::ENull) ||
                its->isEmitter())) {
                /* Encountered an occluder / light source */
                break;
            }

            if (!surface)
                break;

            if (transmittance.isZero())
                return;

            if (its->isMediumTransition())
                medium = its->getTargetMedium(ray.d);

            Vector wo = its->shFrame.toLocal(ray.d);
            BSDFSamplingRecord bRec(*its, -wo, wo, ERadiance);
            bRec.typeMask = BSDF::ENull;
            transmittance *= its->getBSDF()->eval(bRec, EDiscrete);

            ray.o = ray(its->t);
            ray.mint = Epsilon;
            its = &its2;

            if (++interactions > 100) { /// Just a precaution..
                Log(EWarn, "rayIntersectAndLookForEmitter(): round-off error issues?");
                return;
            }
        }

        if (surface) {
            /* Intersected something - check if it was a luminaire */
            if (its->isEmitter()) {
                dRec.setQuery(ray, *its);
                value = transmittance * its->Le(-ray.d);
            }
        } else {
            /* Intersected nothing -- perhaps there is an environment map? */
            const Emitter *env = scene->getEnvironmentEmitter();

            if (env && env->fillDirectSamplingRecord(dRec, ray)) {
                value = transmittance * env->evalEnvironment(RayDifferential(ray));
                dRec.dist = std::numeric_limits<Float>::infinity();
                its->t = std::numeric_limits<Float>::infinity();
            }
        }
    }

    Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA; pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "GuidedPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

private:
    /// The datastructure for guiding paths.
    std::unique_ptr<STree> m_sdTree;

    /// The squared values of our currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_squaredImage;
    /// The currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_image;

    std::vector<ref<Bitmap>> m_images;
    std::vector<Float> m_variances;

    /// This contains the currently estimated variance.
    mutable ref<Film> m_varianceBuffer;

    /// The modes of NEE which are supported.
    enum ENee {
        ENever,
        EKickstart,
        EAlways,
    };

    /**
        How to perform next event estimation (NEE). The following values are valid:
        - "never":     Never performs NEE.
        - "kickstart": Performs NEE for the first few iterations to initialize
                       the SDTree with good direct illumination estimates.
        - "always":    Always performs NEE.
        Default = "never"
    */
    std::string m_neeStr;
    ENee m_nee;

    /// Whether Li should currently perform NEE (automatically set during rendering based on m_nee).
    bool m_doNee;

    enum EBudget {
        ESpp,
        ESeconds,
    };

    /**
        What type of budget to use. The following values are valid:
        - "spp":     Budget is the number of samples per pixel.
        - "seconds": Budget is a time in seconds.
        Default = "seconds"
    */
    std::string m_budgetStr;
    EBudget m_budgetType;
    Float m_budget;

    bool m_isBuilt = false;
    int m_iter;
    bool m_isFinalIter = false;

    int m_sppPerPass;

    int m_passesRendered;
    int m_passesRenderedThisIter;
    mutable std::unique_ptr<ProgressReporter> m_progress;

    std::vector<ref<BlockedRenderProcess>> m_renderProcesses;

    /**
        How to combine the samples from all path-guiding iterations:
        - "discard":    Discard all but the last iteration.
        - "automatic":  Discard all but the last iteration, but automatically assign an appropriately
                        larger budget to the last [Mueller et al. 2018].
        - "inversevar": Combine samples of the last 4 iterations based on their
                        mean pixel variance [Mueller et al. 2018].
        Default     = "automatic" (for reproducibility)
        Recommended = "inversevar"
    */
    std::string m_sampleCombinationStr;
    ESampleCombination m_sampleCombination;
    

    /// Maximum memory footprint of the SDTree in MB. Stops subdividing once reached. -1 to disable.
    int m_sdTreeMaxMemory;

    /**
        The spatial filter to use when splatting radiance samples into the SDTree.
        The following values are valid:
        - "nearest":    No filtering [Mueller et al. 2017].
        - "stochastic": Stochastic box filter; improves upon Mueller et al. [2017]
                        at nearly no computational cost.
        - "box":        Box filter; improves the quality further at significant
                        additional computational cost.
        Default     = "nearest" (for reproducibility)
        Recommended = "stochastic"
    */
    std::string m_spatialFilterStr;
    ESpatialFilter m_spatialFilter;
    
    /**
        The directional filter to use when splatting radiance samples into the SDTree.
        The following values are valid:
        - "nearest":    No filtering [Mueller et al. 2017].
        - "box":        Box filter; improves upon Mueller et al. [2017]
                        at nearly no computational cost.
        Default     = "nearest" (for reproducibility)
        Recommended = "box"
    */
    std::string m_directionalFilterStr;
    EDirectionalFilter m_directionalFilter;

    /**
        Leaf nodes of the spatial binary tree are subdivided if the number of samples
        they received in the last iteration exceeds c * sqrt(2^k) where c is this value
        and k is the iteration index. The first iteration has k==0.
        Default     = 12000 (for reproducibility)
        Recommended = 4000
    */
    int m_sTreeThreshold;

    /**
        Leaf nodes of the directional quadtree are subdivided if the fraction
        of energy they carry exceeds this value.
        Default = 0.01 (1%)
    */
    Float m_dTreeThreshold;

    /**
        When guiding, we perform MIS with the balance heuristic between the guiding
        distribution and the BSDF, combined with probabilistically choosing one of the
        two sampling methods. This factor controls how often the BSDF is sampled
        vs. how often the guiding distribution is sampled.
        Default = 0.5 (50%)
    */
    Float m_bsdfSamplingFraction;

    /**
        The loss function to use when learning the bsdfSamplingFraction using gradient
        descent, following the theory of Neural Importance Sampling [Mueller et al. 2018].
        The following values are valid:
        - "none":  No learning (uses the fixed `m_bsdfSamplingFraction`).
        - "kl":    Optimizes bsdfSamplingFraction w.r.t. the KL divergence.
        - "var":   Optimizes bsdfSamplingFraction w.r.t. variance.
        Default     = "none" (for reproducibility)
        Recommended = "kl"
    */
    std::string m_bsdfSamplingFractionLossStr;
    EBsdfSamplingFractionLoss m_bsdfSamplingFractionLoss;

    /**
        Whether to dump a binary representation of the SD-Tree to disk after every
        iteration. The dumped SD-Tree can be visualized with the accompanying
        visualizer tool.
        Default = false
    */
    bool m_dumpSDTree;

    /**
        Whether to use Adjoint-based Russian Roulette
        Default = false
    */
    bool m_useRR;


    /// The time at which rendering started.
    std::chrono::steady_clock::time_point m_startTime;

    EGuidingMode m_guidingMode = EGuidingMode::EProduct;
    int m_maxProductAwareBounces;
    Float m_productSamplingFraction;

public:
    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS(GuidedPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GuidedPathTracer, "Guided path tracer");
MTS_NAMESPACE_END
