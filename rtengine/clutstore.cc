#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include <glibmm/fileutils.h>
#include <glibmm/miscutils.h>

#include "clutstore.h"

#include "colortemp.h"
#include "iccstore.h"
#include "imagefloat.h"
#include "opthelper.h"
#include "procparams.h"
#include "rt_math.h"
#include "stdimagesource.h"

#include "rtgui/options.h"

namespace
{

// ---------------------------------------------------------------------------
// Shared SSE helper used by CLUT3D::getRGB
// ---------------------------------------------------------------------------

#if defined(__SSE2__) || defined(RT_SIMDE)
vfloat2 getClutValues(const AlignedBuffer<std::uint16_t>& clut_image, size_t index)
{
    const vint v_values = _mm_loadu_si128(reinterpret_cast<const vint*>(clut_image.data + index));
#ifdef __SSE4_1__
    return {
        _mm_cvtepi32_ps(_mm_cvtepu16_epi32(v_values)),
        _mm_cvtepi32_ps(_mm_cvtepu16_epi32(_mm_srli_si128(v_values, 8)))
    };
#else
    const vint v_mask = _mm_set1_epi32(0x0000FFFF);

    vint v_low = _mm_shuffle_epi32(v_values, _MM_SHUFFLE(1, 0, 1, 0));
    vint v_high = _mm_shuffle_epi32(v_values, _MM_SHUFFLE(3, 2, 3, 2));
    v_low = _mm_shufflelo_epi16(v_low, _MM_SHUFFLE(1, 1, 0, 0));
    v_high = _mm_shufflelo_epi16(v_high, _MM_SHUFFLE(1, 1, 0, 0));
    v_low = _mm_shufflehi_epi16(v_low, _MM_SHUFFLE(3, 3, 2, 2));
    v_high = _mm_shufflehi_epi16(v_high, _MM_SHUFFLE(3, 3, 2, 2));
    v_low = vandm(v_low, v_mask);
    v_high = vandm(v_high, v_mask);

    return {
        _mm_cvtepi32_ps(v_low),
        _mm_cvtepi32_ps(v_high)
    };
#endif
}
#endif

// ---------------------------------------------------------------------------
// HaldCLUT loading helper (image-based: PNG / TIFF)
// ---------------------------------------------------------------------------

bool loadHaldFile(
    const Glib::ustring& filename,
    const Glib::ustring& working_color_space,
    AlignedBuffer<std::uint16_t>& clut_image,
    unsigned int& clut_level
)
{
    rtengine::StdImageSource img_src;

    if (!Glib::file_test(filename, Glib::FILE_TEST_EXISTS) || img_src.load(filename)) {
        return false;
    }

    int fw, fh;
    img_src.getFullSize(fw, fh, TR_NONE);

    bool res = false;

    if (fw == fh) {
        int level = 1;

        while (level * level * level < fw) {
            ++level;
        }

        if (level * level * level == fw && level > 1) {
            clut_level = level;
            res = true;
        }
    }

    if (res) {
        rtengine::ColorTemp curr_wb = img_src.getWB();
        std::unique_ptr<rtengine::Imagefloat> img_float = std::unique_ptr<rtengine::Imagefloat>(new rtengine::Imagefloat(fw, fh));
        const PreviewProps pp(0, 0, fw, fh, 1);

        rtengine::procparams::ColorManagementParams icm;
        icm.workingProfile = working_color_space;

        img_src.getImage(curr_wb, TR_NONE, img_float.get(), pp, rtengine::procparams::ToneCurveParams(), rtengine::procparams::RAWParams());

        if (!working_color_space.empty()) {
            img_src.convertColorSpace(img_float.get(), icm, curr_wb);
        }

        AlignedBuffer<std::uint16_t> image(fw * fh * 4 + 4); // getClutValues() loads one pixel in advance

        std::size_t index = 0;

        for (int y = 0; y < fh; ++y) {
            for (int x = 0; x < fw; ++x) {
                image.data[index] = img_float->r(y, x);
                ++index;
                image.data[index] = img_float->g(y, x);
                ++index;
                image.data[index] = img_float->b(y, x);
                index += 2;
            }
        }

        clut_image.swap(image);
    }

    return res;
}

// ---------------------------------------------------------------------------
// Adaptive Gaussian smoothing for .cube LUT export
// ---------------------------------------------------------------------------

// Apply an adaptive bilateral-style Gaussian to a 3D LUT cube stored as a
// flat array in .cube order (R fastest, B slowest):
//   data[b * N*N + g * N + r]  →  RGB triple in [0, 1].
//
// Algorithm:
//   1. Compute a per-node gradient magnitude via central finite differences
//      on the 3×3 Jacobian (9 partial derivatives: 3 output channels × 3
//      input axes).  Normalise to [0, 1].
//   2. For each node build a weighted average over a cubic neighbourhood of
//      radius ceil(2·sigma):
//        w = w_spatial · w_range · w_adaptive
//      where:
//        w_spatial  = exp(-‖Δnode‖² / 2σ²)          — Gaussian distance
//        w_range    = exp(-‖Δvalue‖² / 2σ_r²)        — bilateral colour similarity
//        w_adaptive = 1 / (1 + 4·grad_norm[neighbour]) — down-weight edges
//   3. Blend: output = original + blend·(smooth − original),
//      where blend = strength · (1 − grad_norm[node]) so that high-gradient
//      nodes are blended less (edge preservation).
//
// sigma        — spatial kernel sigma in LUT-node units.
// strength     — global blend in [0, 1].
void smoothCube3D(std::vector<std::array<float, 3>>& cube, int N,
                  float sigma, float strength)
{
    if (N < 2 || sigma <= 0.f || strength <= 0.f) {
        return;
    }

    const int N2    = N * N;
    const int total = N * N * N;

    auto idx = [N, N2](int r, int g, int b) -> int {
        return b * N2 + g * N + r;
    };
    auto clp = [N](int v) -> int {
        return v < 0 ? 0 : (v >= N ? N - 1 : v);
    };

    // ── 1. Gradient magnitudes ────────────────────────────────────────────────
    std::vector<float> grad(total, 0.f);
    float grad_max = 0.f;

    for (int b = 0; b < N; ++b) {
        for (int g = 0; g < N; ++g) {
            for (int r = 0; r < N; ++r) {
                float sum = 0.f;

                // Three input axes: R (axis 0), G (axis 1), B (axis 2)
                const int steps[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };
                for (int a = 0; a < 3; ++a) {
                    const int* s = steps[a];
                    const auto& lo = cube[idx(clp(r-s[0]), clp(g-s[1]), clp(b-s[2]))];
                    const auto& hi = cube[idx(clp(r+s[0]), clp(g+s[1]), clp(b+s[2]))];

                    // At boundaries the effective step is 1 instead of 2
                    const bool onBound =
                        (a == 0 && (r == 0 || r == N-1)) ||
                        (a == 1 && (g == 0 || g == N-1)) ||
                        (a == 2 && (b == 0 || b == N-1));
                    const float denom = onBound ? 1.f : 2.f;

                    for (int c = 0; c < 3; ++c) {
                        const float d = (hi[c] - lo[c]) / denom;
                        sum += d * d;
                    }
                }

                const float gm = std::sqrt(sum);
                grad[idx(r, g, b)] = gm;
                if (gm > grad_max) { grad_max = gm; }
            }
        }
    }

    if (grad_max > 0.f) {
        for (float& gv : grad) { gv /= grad_max; }
    }

    // ── 2. Precompute kernel parameters ──────────────────────────────────────
    const int   radius          = static_cast<int>(std::ceil(2.f * sigma));
    const float inv_2sigma2     = 1.f / (2.f * sigma * sigma);
    // Range sigma: 0.15 in [0,1] colour space — governs bilateral selectivity
    const float inv_2sigmaR2    = 1.f / (2.f * 0.15f * 0.15f);

    // ── 3. Adaptive bilateral Gaussian smoothing ──────────────────────────────
    std::vector<std::array<float, 3>> output(cube);

    for (int b0 = 0; b0 < N; ++b0) {
        for (int g0 = 0; g0 < N; ++g0) {
            for (int r0 = 0; r0 < N; ++r0) {
                const int   ci     = idx(r0, g0, b0);
                const auto& center = cube[ci];
                const float cgrad  = grad[ci];

                std::array<float, 3> wsum  = {0.f, 0.f, 0.f};
                float                wtot  = 0.f;

                for (int db = -radius; db <= radius; ++db) {
                    for (int dg = -radius; dg <= radius; ++dg) {
                        for (int dr = -radius; dr <= radius; ++dr) {
                            const int ni = idx(clp(r0+dr), clp(g0+dg), clp(b0+db));
                            const auto& nb = cube[ni];

                            // Spatial Gaussian
                            const float dist2 = float(dr*dr + dg*dg + db*db);
                            float w = std::exp(-dist2 * inv_2sigma2);

                            // Bilateral: colour-space similarity
                            float cdiff2 = 0.f;
                            for (int c = 0; c < 3; ++c) {
                                const float d = nb[c] - center[c];
                                cdiff2 += d * d;
                            }
                            w *= std::exp(-cdiff2 * inv_2sigmaR2);

                            // Adaptive: down-weight high-gradient neighbours
                            w *= 1.f / (1.f + 4.f * grad[ni]);

                            for (int c = 0; c < 3; ++c) { wsum[c] += w * nb[c]; }
                            wtot += w;
                        }
                    }
                }

                // Blend: less smoothing where local gradient is high
                const float blend = strength * (1.f - cgrad);
                const float iblend = 1.f - blend;

                for (int c = 0; c < 3; ++c) {
                    output[ci][c] = center[c] * iblend + (wsum[c] / wtot) * blend;
                }
            }
        }
    }

    cube = std::move(output);
}

} // anonymous namespace

// ===========================================================================
// CLUT3D — shared base implementation
// ===========================================================================

rtengine::CLUT3D::operator bool() const
{
    return !clut_image.isEmpty();
}

Glib::ustring rtengine::CLUT3D::getFilename() const
{
    return clut_filename;
}

Glib::ustring rtengine::CLUT3D::getProfile() const
{
    return clut_profile;
}

void rtengine::CLUT3D::getRGB(
    float strength,
    std::size_t line_size,
    const float* r,
    const float* g,
    const float* b,
    float* out_rgbx
) const
{
    const unsigned int level = clut_level;
    const unsigned int level_square = level * level;

#if defined(__SSE2__) || defined(RT_SIMDE)
    const vfloat v_strength = F2V(strength);
#endif

    for (std::size_t column = 0; column < line_size; ++column, ++r, ++g, ++b, out_rgbx += 4) {
        const unsigned int red = std::min(flevel_minus_two, *r * flevel_minus_one);
        const unsigned int green = std::min(flevel_minus_two, *g * flevel_minus_one);
        const unsigned int blue = std::min(flevel_minus_two, *b * flevel_minus_one);

        const unsigned int color = red + green * level + blue * level_square;

#if ! defined(__SSE2__) && ! defined(RT_SIMDE)
        const float re = *r * flevel_minus_one - red;
        const float gr = *g * flevel_minus_one - green;
        const float bl = *b * flevel_minus_one - blue;

        size_t index = color * 4;

        float tmp1[4] ALIGNED16;
        tmp1[0] = intp<float>(re, clut_image.data[index + 4], clut_image.data[index]);
        tmp1[1] = intp<float>(re, clut_image.data[index + 5], clut_image.data[index + 1]);
        tmp1[2] = intp<float>(re, clut_image.data[index + 6], clut_image.data[index + 2]);

        index = (color + level) * 4;

        float tmp2[4] ALIGNED16;
        tmp2[0] = intp<float>(re, clut_image.data[index + 4], clut_image.data[index]);
        tmp2[1] = intp<float>(re, clut_image.data[index + 5], clut_image.data[index + 1]);
        tmp2[2] = intp<float>(re, clut_image.data[index + 6], clut_image.data[index + 2]);

        out_rgbx[0] = intp<float>(gr, tmp2[0], tmp1[0]);
        out_rgbx[1] = intp<float>(gr, tmp2[1], tmp1[1]);
        out_rgbx[2] = intp<float>(gr, tmp2[2], tmp1[2]);

        index = (color + level_square) * 4;

        tmp1[0] = intp<float>(re, clut_image.data[index + 4], clut_image.data[index]);
        tmp1[1] = intp<float>(re, clut_image.data[index + 5], clut_image.data[index + 1]);
        tmp1[2] = intp<float>(re, clut_image.data[index + 6], clut_image.data[index + 2]);

        index = (color + level + level_square) * 4;

        tmp2[0] = intp<float>(re, clut_image.data[index + 4], clut_image.data[index]);
        tmp2[1] = intp<float>(re, clut_image.data[index + 5], clut_image.data[index + 1]);
        tmp2[2] = intp<float>(re, clut_image.data[index + 6], clut_image.data[index + 2]);

        tmp1[0] = intp<float>(gr, tmp2[0], tmp1[0]);
        tmp1[1] = intp<float>(gr, tmp2[1], tmp1[1]);
        tmp1[2] = intp<float>(gr, tmp2[2], tmp1[2]);

        out_rgbx[0] = intp<float>(bl, tmp1[0], out_rgbx[0]);
        out_rgbx[1] = intp<float>(bl, tmp1[1], out_rgbx[1]);
        out_rgbx[2] = intp<float>(bl, tmp1[2], out_rgbx[2]);

        out_rgbx[0] = intp<float>(strength, out_rgbx[0], *r);
        out_rgbx[1] = intp<float>(strength, out_rgbx[1], *g);
        out_rgbx[2] = intp<float>(strength, out_rgbx[2], *b);
#else
        const vfloat v_in = _mm_set_ps(0.0f, *b, *g, *r);
        const vfloat v_tmp = v_in * F2V(flevel_minus_one);
        const vfloat v_rgb = v_tmp - _mm_cvtepi32_ps(_mm_cvttps_epi32(vminf(v_tmp, F2V(flevel_minus_two))));

        size_t index = color * 4;

        const vfloat v_r = PERMUTEPS(v_rgb, _MM_SHUFFLE(0, 0, 0, 0));

        vfloat2 v_clut_values = getClutValues(clut_image, index);
        vfloat v_tmp1 = vintpf(v_r, v_clut_values.y, v_clut_values.x);

        index = (color + level) * 4;

        v_clut_values = getClutValues(clut_image, index);
        vfloat v_tmp2 = vintpf(v_r, v_clut_values.y, v_clut_values.x);

        const vfloat v_g = PERMUTEPS(v_rgb, _MM_SHUFFLE(1, 1, 1, 1));

        vfloat v_out = vintpf(v_g, v_tmp2, v_tmp1);

        index = (color + level_square) * 4;

        v_clut_values = getClutValues(clut_image, index);
        v_tmp1 = vintpf(v_r, v_clut_values.y, v_clut_values.x);

        index = (color + level + level_square) * 4;

        v_clut_values = getClutValues(clut_image, index);
        v_tmp2 = vintpf(v_r, v_clut_values.y, v_clut_values.x);

        v_tmp1 = vintpf(v_g, v_tmp2, v_tmp1);

        const vfloat v_b = PERMUTEPS(v_rgb, _MM_SHUFFLE(2, 2, 2, 2));

        v_out = vintpf(v_b, v_tmp1, v_out);

        STVF(*out_rgbx, vintpf(v_strength, v_out, v_in));
#endif
    }
}

// ===========================================================================
// HaldCLUT — image-based (PNG / TIFF) Hald CLUT
// ===========================================================================

bool rtengine::HaldCLUT::load(const Glib::ustring& filename)
{
    if (loadHaldFile(filename, "", clut_image, clut_level)) {
        Glib::ustring name, ext;
        splitClutFilename(filename, name, ext, clut_profile);

        clut_filename = filename;
        clut_level *= clut_level;
        flevel_minus_one = static_cast<float>(clut_level - 1) / 65535.0f;
        flevel_minus_two = static_cast<float>(clut_level - 2);
        return true;
    }

    return false;
}

Glib::ustring rtengine::HaldCLUT::createIdentityTempFile(int level)
{
    const int cube  = level * level;          // samples per axis
    const int size  = level * level * level;  // image side length
    const float den = static_cast<float>(cube - 1);

    Imagefloat img(size, size);

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int idx   = y * size + x;
            const int b_idx = idx / (cube * cube);
            const int rem   = idx % (cube * cube);
            const int g_idx = rem / cube;
            const int r_idx = rem % cube;

            img.r(y, x) = r_idx * 65535.0f / den;
            img.g(y, x) = g_idx * 65535.0f / den;
            img.b(y, x) = b_idx * 65535.0f / den;
        }
    }

    const Glib::ustring tmpPath =
        Glib::build_filename(Glib::get_tmp_dir(), "rt_hald_identity.png");

    if (img.saveAsPNG(tmpPath, 16) != 0) {
        return {};
    }

    return tmpPath;
}

void rtengine::HaldCLUT::splitClutFilename(
    const Glib::ustring& filename,
    Glib::ustring& name,
    Glib::ustring& extension,
    Glib::ustring& profile_name,
    bool checkProfile
)
{
    Glib::ustring basename = Glib::path_get_basename(filename);

    const Glib::ustring::size_type last_dot_pos = basename.rfind('.');

    if (last_dot_pos != Glib::ustring::npos) {
        name.assign(basename, 0, last_dot_pos);
        extension.assign(basename, last_dot_pos + 1, Glib::ustring::npos);
    } else {
        name = basename;
    }

    if (checkProfile) {
        profile_name = "sRGB";

        if (!name.empty()) {
            for (const auto& working_profile : rtengine::ICCStore::getInstance()->getWorkingProfiles()) {
                if (
                    !working_profile.empty()
                    && std::search(name.rbegin(), name.rend(), working_profile.rbegin(), working_profile.rend()) == name.rbegin()
                ) {
                    profile_name = working_profile;
                    name.erase(name.size() - working_profile.size());
                    break;
                }
            }
        }
    }
}

// ===========================================================================
// CubeLUT — text-based .cube 3D LUT (Adobe / DaVinci Resolve format)
// ===========================================================================

bool rtengine::CubeLUT::load(const Glib::ustring& filename)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        return false;
    }

    int size = 0;
    float domain_min[3] = {0.f, 0.f, 0.f};
    float domain_max[3] = {1.f, 1.f, 1.f};
    std::vector<std::array<float, 3>> entries;

    std::string line;
    while (std::getline(file, line)) {
        // Strip Windows-style carriage return
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        if (line.rfind("LUT_3D_SIZE", 0) == 0) {
            std::istringstream ss(line.substr(11));
            ss >> size;
        } else if (line.rfind("DOMAIN_MIN", 0) == 0) {
            std::istringstream ss(line.substr(10));
            ss >> domain_min[0] >> domain_min[1] >> domain_min[2];
        } else if (line.rfind("DOMAIN_MAX", 0) == 0) {
            std::istringstream ss(line.substr(10));
            ss >> domain_max[0] >> domain_max[1] >> domain_max[2];
        } else if (line.rfind("TITLE", 0) == 0
                   || line.rfind("LUT_1D_SIZE", 0) == 0
                   || line.rfind("LUT_1D_INPUT_TABLE", 0) == 0
                   || line.rfind("LUT_3D_INPUT_TABLE", 0) == 0) {
            // header-only keywords — nothing to do
        } else {
            float r, g, b;
            std::istringstream ss(line);
            if (ss >> r >> g >> b) {
                entries.push_back({r, g, b});
            }
        }
    }

    if (size <= 1 || static_cast<int>(entries.size()) != size * size * size) {
        return false;
    }

    clut_level = size;

    const int total = size * size * size;
    AlignedBuffer<std::uint16_t> image(total * 4 + 4); // +4: getRGB reads one pixel ahead

    for (int i = 0; i < total; ++i) {
        for (int c = 0; c < 3; ++c) {
            const float range = domain_max[c] - domain_min[c];
            float v = (range > 0.f) ? (entries[i][c] - domain_min[c]) / range : 0.f;
            v = std::max(0.f, std::min(1.f, v));
            image.data[i * 4 + c] = static_cast<std::uint16_t>(v * 65535.f + 0.5f);
        }
        image.data[i * 4 + 3] = 0;
    }

    clut_image.swap(image);

    // Determine colour profile from filename suffix (same convention as HaldCLUT)
    Glib::ustring name, ext;
    HaldCLUT::splitClutFilename(filename, name, ext, clut_profile);

    clut_filename = filename;
    flevel_minus_one = static_cast<float>(clut_level - 1) / 65535.0f;
    flevel_minus_two = static_cast<float>(clut_level - 2);

    return true;
}

Glib::ustring rtengine::CubeLUT::createIdentityTempFile(int size)
{
    if (size < 2) {
        return {};
    }

    // Layout: width = size*size, height = size.
    // Pixel (y=b, x=g*size+r) encodes input colour (r/(size-1), g/(size-1), b/(size-1)).
    const float den = static_cast<float>(size - 1);
    Imagefloat img(size * size, size);

    for (int b = 0; b < size; ++b) {
        for (int g = 0; g < size; ++g) {
            for (int r = 0; r < size; ++r) {
                const int x = g * size + r;
                img.r(b, x) = r * 65535.f / den;
                img.g(b, x) = g * 65535.f / den;
                img.b(b, x) = b * 65535.f / den;
            }
        }
    }

    const Glib::ustring tmpPath =
        Glib::build_filename(Glib::get_tmp_dir(), "rt_cube_identity.png");

    if (img.saveAsPNG(tmpPath, 16) != 0) {
        return {};
    }

    return tmpPath;
}

bool rtengine::CubeLUT::saveAsCubeFile(const IImagefloat* img, int size,
                                        const Glib::ustring& destPath,
                                        const CubeLUTSmoothParams& smooth)
{
    // ── Extract cube to a flat float array in [0, 1] ─────────────────────────
    // Layout matches .cube order: index = b*N² + g*N + r  (R fastest).
    // Image layout: pixel (y=b, x=g*size+r).
    const float scale = 1.f / 65535.f;
    const int total   = size * size * size;

    std::vector<std::array<float, 3>> cube(total);
    for (int b = 0; b < size; ++b) {
        for (int g = 0; g < size; ++g) {
            for (int r = 0; r < size; ++r) {
                const int x   = g * size + r;
                const int idx = b * size * size + g * size + r;
                cube[idx][0] = std::max(0.f, std::min(1.f, img->r(b, x) * scale));
                cube[idx][1] = std::max(0.f, std::min(1.f, img->g(b, x) * scale));
                cube[idx][2] = std::max(0.f, std::min(1.f, img->b(b, x) * scale));
            }
        }
    }

    // ── Optional adaptive Gaussian smoothing ─────────────────────────────────
    smoothCube3D(cube, size, smooth.sigma, smooth.strength);

    // ── Write .cube file ──────────────────────────────────────────────────────
    std::ofstream file(destPath.c_str());
    if (!file.is_open()) {
        return false;
    }

    file << "TITLE \"RawTherapee Export\"\n\n";
    file << "LUT_3D_SIZE " << size << "\n\n";
    file << "DOMAIN_MIN 0.0 0.0 0.0\n";
    file << "DOMAIN_MAX 1.0 1.0 1.0\n\n";
    file << std::fixed << std::setprecision(6);

    for (int idx = 0; idx < total; ++idx) {
        file << cube[idx][0] << ' ' << cube[idx][1] << ' ' << cube[idx][2] << '\n';
    }

    return file.good();
}

// ===========================================================================
// CLUTStore — factory + LRU cache
// ===========================================================================

rtengine::CLUTStore& rtengine::CLUTStore::getInstance()
{
    static CLUTStore instance;
    return instance;
}

std::shared_ptr<rtengine::CLUT3D> rtengine::CLUTStore::getClut(const Glib::ustring& filename) const
{
    std::shared_ptr<rtengine::CLUT3D> result;

    const Glib::ustring full_filename =
        !Glib::path_is_absolute(filename)
            ? Glib::ustring(Glib::build_filename(App::get().options().clutsDir, filename))
            : filename;

    if (!cache.get(full_filename, result)) {
        // Choose concrete class from file extension
        Glib::ustring name, ext, dummy;
        HaldCLUT::splitClutFilename(full_filename, name, ext, dummy, false);
        ext = ext.casefold();

        std::unique_ptr<CLUT3D> clut;
        if (ext == "cube") {
            clut.reset(new CubeLUT());
        } else {
            clut.reset(new HaldCLUT());
        }

        if (clut->load(full_filename)) {
            result = std::move(clut);
            cache.insert(full_filename, result);
        }
    }

    return result;
}

void rtengine::CLUTStore::clearCache()
{
    cache.clear();
}

rtengine::CLUTStore::CLUTStore() :
    cache(App::get().options().clutCacheSize)
{
}
