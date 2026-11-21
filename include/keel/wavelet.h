#pragma once

#include "keel/array.h"
#include "keel/core.h"
#include "keel/math.h"
#include "keel/memory.h"
#include "keel/ref.h"
#include "keel/string.h"

namespace keel::wavelet {

// ============================================================================
// Filter Banks (haar, db1..db10, sym2..sym10, coif1..coif5, bior*, rbio*)
// ============================================================================

struct FilterBank {
    Array<f64> lo_d; ///< Analysis (decomposition) low-pass filter
    Array<f64> hi_d; ///< Analysis (decomposition) high-pass filter
    Array<f64> lo_r; ///< Synthesis (reconstruction) low-pass filter
    Array<f64> hi_r; ///< Synthesis (reconstruction) high-pass filter
};

/// Look up a built-in orthogonal or biorthogonal wavelet filter bank by name.
/// Returns `true` if found, `false` otherwise.
[[nodiscard]] bool lookup_filters(StringRef name, FilterBank& out, Allocator* alloc = nullptr);

/// Construct an orthogonal filter quadruple [Lo_D, Hi_D, Lo_R, Hi_R] from a
/// unit-sum scaling filter `scaling_w`.
[[nodiscard]] FilterBank orthfilt(ArrayRef<f64> scaling_w, Allocator* alloc = nullptr);

/// Quadrature mirror filter: `out[k] = (-1)^(k + parity) * x[N - 1 - k]`.
void qmf(ArrayRef<f64> x, MutArrayRef<f64> out, i32 parity = 0) noexcept;

// ============================================================================
// 1D Single-Level & Multilevel DWT (symmetric boundary extension)
// ============================================================================

struct Dwt1Result {
    Array<f64> ca; ///< Approximation coefficients
    Array<f64> cd; ///< Detail coefficients
};

[[nodiscard]] Dwt1Result dwt(ArrayRef<f64> x, ArrayRef<f64> lo_d, ArrayRef<f64> hi_d, Allocator* alloc = nullptr);

[[nodiscard]] Array<f64> idwt(ArrayRef<f64> ca, ArrayRef<f64> cd, ArrayRef<f64> lo_r, ArrayRef<f64> hi_r,
                              i64 target_len = -1, Allocator* alloc = nullptr);

struct Wavedec1Result {
    Array<f64> c;   ///< Packed [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    Array<usize> l; ///< Bookkeeping lengths [|cA_n|, |cD_n|, ..., |cD_1|, |x|]
};

[[nodiscard]] Wavedec1Result wavedec(ArrayRef<f64> x, i32 level, const FilterBank& fb, Allocator* alloc = nullptr);

[[nodiscard]] Array<f64> waverec(ArrayRef<f64> c, ArrayRef<usize> l, const FilterBank& fb, Allocator* alloc = nullptr);

[[nodiscard]] Array<f64> appcoef(ArrayRef<f64> c, ArrayRef<usize> l, ArrayRef<f64> lo_r, ArrayRef<f64> hi_r,
                                 i32 level, Allocator* alloc = nullptr);

[[nodiscard]] Array<f64> detcoef(ArrayRef<f64> c, ArrayRef<usize> l, i32 level, Allocator* alloc = nullptr);

// ============================================================================
// 2D Separable DWT / IDWT (column-major matrices)
// ============================================================================

struct Dwt2Result {
    Array<f64> ca; ///< LL (out_rows x out_cols, column-major)
    Array<f64> ch; ///< LH horizontal detail
    Array<f64> cv; ///< HL vertical detail
    Array<f64> cd; ///< HH diagonal detail
    usize out_rows = 0;
    usize out_cols = 0;
};

[[nodiscard]] Dwt2Result dwt2(ArrayRef<f64> x_colmajor, usize rows, usize cols, const FilterBank& fb,
                              Allocator* alloc = nullptr);

struct Idwt2Result {
    Array<f64> out; ///< Column-major matrix (rows x cols)
    usize rows = 0;
    usize cols = 0;
};

[[nodiscard]] Idwt2Result idwt2(ArrayRef<f64> ca, ArrayRef<f64> ch, ArrayRef<f64> cv, ArrayRef<f64> cd, usize rows,
                                usize cols, const FilterBank& fb, i64 target_rows = -1, i64 target_cols = -1,
                                Allocator* alloc = nullptr);

// ============================================================================
// Undecimated Transforms: SWT & MODWT (column-major)
// ============================================================================

struct SwtResult {
    Array<f64> swa_colmajor; ///< Approximations (levels x length, column-major)
    Array<f64> swd_colmajor; ///< Details (levels x length, column-major)
    usize levels = 0;
    usize length = 0;
};

/// Stationary (a trous) wavelet transform. Signal length must be divisible by 2^level.
[[nodiscard]] SwtResult swt_ad(ArrayRef<f64> x, i32 level, ArrayRef<f64> lo_d, ArrayRef<f64> hi_d,
                               Allocator* alloc = nullptr);

/// Maximal-overlap DWT (MODWT), column-major (level + 1) x N:
/// rows 0..level-1 = wavelet bands W_1..W_level, row level = scaling band V_level.
[[nodiscard]] Array<f64> modwt(ArrayRef<f64> x, i32 level, const FilterBank& fb, Allocator* alloc = nullptr);

/// Inverse MODWT from column-major (levels + 1) x n matrix `w_colmajor`.
[[nodiscard]] Array<f64> imodwt(ArrayRef<f64> w_colmajor, usize levels, usize n, const FilterBank& fb,
                                Allocator* alloc = nullptr);

// ============================================================================
// Thresholding, Noise Estimation, Denoising & Entropy
// ============================================================================

void threshold_soft(ArrayRef<f64> x, MutArrayRef<f64> out, f64 t) noexcept;
void threshold_hard(ArrayRef<f64> x, MutArrayRef<f64> out, f64 t) noexcept;

/// Robust noise standard deviation estimate: median(|detail|) / 0.6745.
[[nodiscard]] f64 noise_sigma_mad(ArrayRef<f64> detail, Allocator* alloc = nullptr);

/// Universal-threshold (VisuShrink) soft-thresholding 1D wavelet denoiser.
[[nodiscard]] Array<f64> denoise_visushrink(ArrayRef<f64> x, i32 level, const FilterBank& fb,
                                            Allocator* alloc = nullptr);

enum class EntropyKind : u8 {
    Shannon = 0,
    LogEnergy,
    Threshold,
    Sure,
    Norm,
};

[[nodiscard]] f64 entropy(ArrayRef<f64> x, EntropyKind kind, f64 param = 0.0) noexcept;

} // namespace keel::wavelet
