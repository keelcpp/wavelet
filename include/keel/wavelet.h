#pragma once

#include "keel/array.h"
#include "keel/complex.h"
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

// ============================================================================
// Multilevel 1D/2D Utilities, Haar & Dyadic Primitives
// ============================================================================

/// Energy percentages for a 1D `wavedec` decomposition `(c, l)`.
/// Writes approximation percentage to `ea_out` and finest-first detail percentages
/// `Ed[0..n_levels-1]` (level 1 .. level N) into `ed_out`.
[[nodiscard]] bool wenergy(ArrayRef<f64> c, ArrayRef<usize> l, f64& ea_out, MutArrayRef<f64> ed_out) noexcept;

/// Direct upsampling and synthesis filtering (`is_detail == false` for approximation, `true` for detail).
[[nodiscard]] Array<f64> upcoef(bool is_detail, ArrayRef<f64> x, const FilterBank& fb, i32 level,
                                i64 target_len = -1, Allocator* alloc = nullptr);

/// Single-branch 1D reconstruction from a `wavedec` `(c, l)` decomposition.
/// `level == -1` selects the coarsest level `N`.
[[nodiscard]] Array<f64> wrcoef(bool is_detail, ArrayRef<f64> c, ArrayRef<usize> l, ArrayRef<f64> lo_r,
                                ArrayRef<f64> hi_r, i32 level = -1, Allocator* alloc = nullptr);

/// Maximum useful wavelet decomposition level for a signal of length `signal_len` and filter length `filter_len`.
[[nodiscard]] usize wmaxlev(usize signal_len, usize filter_len) noexcept;

enum class Detail2D : u8 {
    Horizontal = 0,
    Vertical,
    Diagonal,
};

struct Wavedec2Result {
    Array<f64> c;        ///< Packed [cA_N | cH_N cV_N cD_N | ... | cH_1 cV_1 cD_1]
    Array<usize> s_rows; ///< Bookkeeping row sizes (N + 2 entries)
    Array<usize> s_cols; ///< Bookkeeping col sizes (N + 2 entries)
};

[[nodiscard]] Wavedec2Result wavedec2(ArrayRef<f64> x_colmajor, usize rows, usize cols, i32 level,
                                      const FilterBank& fb, Allocator* alloc = nullptr);

[[nodiscard]] Idwt2Result appcoef2(ArrayRef<f64> c, ArrayRef<usize> s_rows, ArrayRef<usize> s_cols,
                                   const FilterBank& fb, i32 level = -1, Allocator* alloc = nullptr);

[[nodiscard]] Idwt2Result waverec2(ArrayRef<f64> c, ArrayRef<usize> s_rows, ArrayRef<usize> s_cols,
                                   const FilterBank& fb, Allocator* alloc = nullptr);

[[nodiscard]] Idwt2Result detcoef2(Detail2D kind, ArrayRef<f64> c, ArrayRef<usize> s_rows, ArrayRef<usize> s_cols,
                                   i32 level, Allocator* alloc = nullptr);

/// Largest k such that 2^k divides `n`.
[[nodiscard]] i32 haar_max_level(usize n) noexcept;

/// Single-level 1D Haar decomposition (orthogonal or lifting integer mode).
/// Requires `x.size()` even, `a_out.size() >= x.size()/2`, `d_out.size() >= x.size()/2`.
void haar_step(ArrayRef<f64> x, MutArrayRef<f64> a_out, MutArrayRef<f64> d_out,
               bool integer_lifting = false) noexcept;

void haar_step_c64(ArrayRef<c64> x, MutArrayRef<c64> a_out, MutArrayRef<c64> d_out) noexcept;

/// Single-level 1D inverse Haar reconstruction (orthogonal or lifting integer mode).
/// Requires `a.size() == d.size()` and `x_out.size() >= 2 * a.size()`.
void ihaar_step(ArrayRef<f64> a, ArrayRef<f64> d, MutArrayRef<f64> x_out,
                bool integer_lifting = false) noexcept;

void ihaar_step_c64(ArrayRef<c64> a, ArrayRef<f64> d, MutArrayRef<c64> x_out) noexcept;
void ihaar_step_c64(ArrayRef<c64> a, ArrayRef<c64> d, MutArrayRef<c64> x_out) noexcept;

/// 1D dyadic downsampling (`keep_odd == false` keeps 0-based indices 1,3,...; `true` keeps 0,2,...).
void dyaddown_1d(ArrayRef<f64> x, MutArrayRef<f64> out, bool keep_odd = false) noexcept;

/// 1D dyadic upsampling (`pad_ends == false` -> length `2*N-1`; `true` -> length `2*N+1` with leading/trailing 0).
void dyadup_1d(ArrayRef<f64> x, MutArrayRef<f64> out, bool pad_ends = false) noexcept;

// ============================================================================
// Continuous Wavelet Shapes, Center Frequency & Continuous Wavelet Transform
// ============================================================================

void mexihat(f64 lb, f64 ub, MutArrayRef<f64> psi, MutArrayRef<f64> x) noexcept;
void morlet(f64 lb, f64 ub, MutArrayRef<f64> psi, MutArrayRef<f64> x) noexcept;
[[nodiscard]] f64 meyeraux(f64 v) noexcept;
void meyeraux(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept;
void shanwavf(f64 lb, f64 ub, f64 fb, f64 fc, MutArrayRef<c64> psi, MutArrayRef<f64> x) noexcept;
void cmorwavf(f64 lb, f64 ub, f64 fb, f64 fc, MutArrayRef<c64> psi, MutArrayRef<f64> x) noexcept;
void fbspwavf(f64 lb, f64 ub, i32 m, f64 fb, f64 fc, MutArrayRef<c64> psi, MutArrayRef<f64> x) noexcept;
void gauswavf(f64 lb, f64 ub, i32 p, MutArrayRef<f64> psi, MutArrayRef<f64> x) noexcept;
void cgauwavf(f64 lb, f64 ub, i32 p, MutArrayRef<c64> psi, MutArrayRef<f64> x, Allocator* alloc = nullptr);

struct WavefunPsiResult {
    Array<f64> psi_real;
    Array<c64> psi_cplx;
    Array<f64> xval;
    bool is_complex = false;
    bool ok = false;
};

[[nodiscard]] WavefunPsiResult wavefun_psi(StringRef name, i32 iter = 8, Allocator* alloc = nullptr);

struct CentfrqResult {
    f64 freq = 0.0;
    Array<f64> xval;
    Array<f64> recfreq_real;
    Array<c64> recfreq_cplx;
    bool is_complex = false;
    bool ok = false;
};

[[nodiscard]] CentfrqResult centfrq(StringRef name, i32 iter = 8, bool with_recfreq = false,
                                    Allocator* alloc = nullptr);

enum class CwtWavelet : u8 {
    Morse = 0,
    Amor,
    Bump,
};

struct CwtResult {
    Array<c64> cfs_colmajor; ///< Complex CWT coefficients (num_scales x num_samples, column-major)
    Array<f64> freqs;        ///< Scale frequencies in Hz (length num_scales)
    Array<f64> coi;          ///< Cone of influence frequencies in Hz (length num_samples)
    usize num_scales = 0;
    usize num_samples = 0;
};

/// 1D analytic Continuous Wavelet Transform (Morse, Analytic Morlet, or Bump).
/// Pass `fmin > 0 && fmax > fmin` to restrict frequency limits; leave `0.0` for automatic range.
[[nodiscard]] CwtResult cwt(ArrayRef<f64> x, CwtWavelet wavelet, f64 sample_rate,
                            i32 voices_per_octave = 10, f64 time_bandwidth = 60.0, f64 gamma = 3.0,
                            bool extend_signal = true, f64 fmin = 0.0, f64 fmax = 0.0,
                            Allocator* alloc = nullptr);

// ============================================================================
// 1-D Signal Boundary Extension (wextend)
// ============================================================================

enum class ExtendMode : u8 {
    kSym = 0,   // half-point symmetric ('sym', 'symh')
    kSymW,      // whole-point symmetric ('symw')
    kAsym,      // half-point antisymmetric ('asym', 'asymh')
    kAsymW,     // whole-point antisymmetric ('asymw')
    kSp0,       // order-0 spline (edge replication)
    kSp1,       // order-1 spline (linear extrapolation: 'sp1', 'spd')
    kPer,       // periodic with edge-pad on odd length ('per')
    kZpd,       // zero padding ('zpd')
    kPpd,       // pure periodic ('ppd')
};

enum class ExtendSide : u8 {
    kBoth = 0,
    kLeft,
    kRight,
};

[[nodiscard]] Array<f64> wextend_1d(ArrayRef<f64> x, i64 lf, ExtendSide side, ExtendMode mode,
                                    Allocator* alloc = nullptr);
[[nodiscard]] Array<c64> wextend_1d_c64(ArrayRef<c64> x, i64 lf, ExtendSide side, ExtendMode mode,
                                        Allocator* alloc = nullptr);

} // namespace keel::wavelet
