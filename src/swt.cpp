#include "keel/wavelet.h"

#include "keel/arena.h"

namespace keel::wavelet {

namespace {

[[nodiscard]] usize add_mod(usize a, usize b, usize n) noexcept {
    return (a >= n - b) ? (a - (n - b)) : (a + b);
}

[[nodiscard]] usize sub_mod(usize a, usize b, usize n) noexcept {
    return (a >= b) ? (a - b) : (a + (n - b));
}

Array<f64> circ_corr_fwd(ArrayRef<f64> x, ArrayRef<f64> h, usize step_mod, Allocator* alloc) {
    const usize n = x.size();
    Array<f64> y(alloc);
    y.resize(n);
    for (usize i = 0; i < n; ++i) y[i] = 0.0;
    if (n == 0 || h.empty()) return y;
    for (usize i = 0; i < n; ++i) {
        f64 acc = 0.0;
        usize idx = i;
        for (usize j = 0; j < h.size(); ++j) {
            acc += h[j] * x[idx];
            idx = add_mod(idx, step_mod, n);
        }
        y[i] = acc;
    }
    return y;
}

Array<f64> circ_corr_back(ArrayRef<f64> x, ArrayRef<f64> h, usize step_mod, Allocator* alloc) {
    const usize n = x.size();
    Array<f64> y(alloc);
    y.resize(n);
    for (usize i = 0; i < n; ++i) y[i] = 0.0;
    if (n == 0 || h.empty()) return y;
    for (usize i = 0; i < n; ++i) {
        f64 acc = 0.0;
        usize idx = i;
        for (usize j = 0; j < h.size(); ++j) {
            acc += h[j] * x[idx];
            idx = sub_mod(idx, step_mod, n);
        }
        y[i] = acc;
    }
    return y;
}

} // namespace

SwtResult swt_ad(ArrayRef<f64> x, i32 level, ArrayRef<f64> lo_d, ArrayRef<f64> hi_d, Allocator* alloc) {
    SwtResult res{Array<f64>(alloc), Array<f64>(alloc), 0, 0};
    const usize lf = lo_d.size();
    if (level < 1 || lf < 2 || (lf & 1u) != 0u || lf != hi_d.size()) return res;

    const usize n = x.size();
    const usize h_rows = as<usize>(level);
    if (n == 0 || h_rows >= sizeof(usize) * 8) return res;

    const usize needed = as<usize>(1) << h_rows;
    if (n % needed != 0) return res;
    if (h_rows > Limits<usize>::kMax / n) return res;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> xv(scratch.allocator());
    xv.resize(n);
    for (usize i = 0; i < n; ++i) xv[i] = x[i];

    Array<f64> a_row(scratch.allocator());
    Array<f64> d_row(scratch.allocator());
    a_row.resize(n);
    d_row.resize(n);

    res.swa_colmajor.resize(h_rows * n);
    res.swd_colmajor.resize(h_rows * n);
    res.levels = h_rows;
    res.length = n;

    const usize half = lf / 2;

    auto conv_dilated = [&](ArrayRef<f64> src, ArrayRef<f64> h, usize step, MutArrayRef<f64> out) noexcept {
        for (usize m = 0; m < n; ++m) out[m] = 0.0;
        // Initial offset = (half * step) mod n, computed without overflow.
        usize off = 0;
        for (usize r = 0; r < half; ++r) {
            off = add_mod(off, step, n);
        }
        for (usize t = 0; t < lf; ++t) {
            const f64 ht = h[t];
            if (ht != 0.0) {
                usize idx = off;
                for (usize m = 0; m < n; ++m) {
                    out[m] += ht * src[idx];
                    if (++idx == n) idx = 0;
                }
            }
            off = sub_mod(off, step, n);
        }
    };

    for (usize k = 0; k < h_rows; ++k) {
        const usize step = as<usize>(1) << k;
        conv_dilated(ArrayRef<f64>(xv.data(), n), hi_d, step, MutArrayRef<f64>(d_row.data(), n));
        conv_dilated(ArrayRef<f64>(xv.data(), n), lo_d, step, MutArrayRef<f64>(a_row.data(), n));
        for (usize c = 0; c < n; ++c) {
            res.swa_colmajor[c * h_rows + k] = a_row[c];
            res.swd_colmajor[c * h_rows + k] = d_row[c];
        }
        for (usize c = 0; c < n; ++c) xv[c] = a_row[c];
    }
    return res;
}

Array<f64> modwt(ArrayRef<f64> x, i32 level, const FilterBank& fb, Allocator* alloc) {
    Array<f64> out(alloc);
    const usize lf = fb.lo_d.size();
    if (level < 1 || x.empty() || lf < 2 || fb.hi_d.size() != lf) return out;

    const usize n = x.size();
    const usize lv = as<usize>(level);
    const usize h_rows = lv + 1;
    if (h_rows > Limits<usize>::kMax / n) return out;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> h_lo(scratch.allocator());
    Array<f64> h_hi(scratch.allocator());
    h_lo.resize(lf);
    h_hi.resize(lf);
    for (usize i = 0; i < lf; ++i) {
        h_lo[i] = fb.lo_d[lf - 1 - i] * kInvSqrt2;
        h_hi[i] = fb.hi_d[lf - 1 - i] * kInvSqrt2;
    }

    Array<f64> a(scratch.allocator());
    a.resize(n);
    for (usize i = 0; i < n; ++i) a[i] = x[i];

    out.resize(h_rows * n);
    usize step_mod = 1 % n;
    for (usize k = 0; k < lv; ++k) {
        Array<f64> a_next =
            circ_corr_back(ArrayRef<f64>(a.data(), n), ArrayRef<f64>(h_lo.data(), lf), step_mod, scratch.allocator());
        Array<f64> d_next =
            circ_corr_back(ArrayRef<f64>(a.data(), n), ArrayRef<f64>(h_hi.data(), lf), step_mod, scratch.allocator());
        for (usize c = 0; c < n; ++c) out[c * h_rows + k] = d_next[c];
        a = move(a_next);
        step_mod = add_mod(step_mod, step_mod, n);
    }
    for (usize c = 0; c < n; ++c) out[c * h_rows + lv] = a[c];
    return out;
}

Array<f64> imodwt(ArrayRef<f64> w_colmajor, usize levels, usize n, const FilterBank& fb, Allocator* alloc) {
    Array<f64> a(alloc);
    const usize lf = fb.lo_d.size();
    if (levels == 0 || levels == Limits<usize>::kMax || n == 0 || lf < 2 || fb.hi_d.size() != lf) return a;

    const usize h_rows = levels + 1;
    if (h_rows > Limits<usize>::kMax / n || w_colmajor.size() < h_rows * n) return a;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> h_lo(scratch.allocator());
    Array<f64> h_hi(scratch.allocator());
    h_lo.resize(lf);
    h_hi.resize(lf);
    for (usize i = 0; i < lf; ++i) {
        h_lo[i] = fb.lo_d[lf - 1 - i] * kInvSqrt2;
        h_hi[i] = fb.hi_d[lf - 1 - i] * kInvSqrt2;
    }

    // Precompute step_mod[k] = (2^k) mod n without shift UB or overflow.
    Array<usize> steps(scratch.allocator());
    steps.resize(levels);
    usize cur_step = 1 % n;
    for (usize k = 0; k < levels; ++k) {
        steps[k] = cur_step;
        cur_step = add_mod(cur_step, cur_step, n);
    }

    a.resize(n);
    for (usize c = 0; c < n; ++c) a[c] = w_colmajor[c * h_rows + levels];

    Array<f64> d(scratch.allocator());
    d.resize(n);
    for (usize kk = levels; kk > 0; --kk) {
        const usize k = kk - 1;
        const usize step_mod = steps[k];
        for (usize c = 0; c < n; ++c) d[c] = w_colmajor[c * h_rows + k];
        Array<f64> lo =
            circ_corr_fwd(ArrayRef<f64>(a.data(), n), ArrayRef<f64>(h_lo.data(), lf), step_mod, scratch.allocator());
        Array<f64> hi =
            circ_corr_fwd(ArrayRef<f64>(d.data(), n), ArrayRef<f64>(h_hi.data(), lf), step_mod, scratch.allocator());
        for (usize c = 0; c < n; ++c) a[c] = lo[c] + hi[c];
    }
    return a;
}

} // namespace keel::wavelet
