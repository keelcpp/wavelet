#include "keel/wavelet.h"

namespace keel::wavelet {

namespace {

Array<f64> circ_corr_fwd(ArrayRef<f64> x, ArrayRef<f64> h, usize step, Allocator* alloc) {
    const usize n = x.size();
    Array<f64> y(alloc);
    y.resize(n);
    for (usize i = 0; i < n; ++i) y[i] = 0.0;
    if (n == 0 || h.empty()) return y;
    for (usize i = 0; i < n; ++i) {
        f64 acc = 0.0;
        for (usize j = 0; j < h.size(); ++j) {
            acc += h[j] * x[(i + j * step) % n];
        }
        y[i] = acc;
    }
    return y;
}

Array<f64> circ_corr_back(ArrayRef<f64> x, ArrayRef<f64> h, usize step, Allocator* alloc) {
    const usize n = x.size();
    Array<f64> y(alloc);
    y.resize(n);
    for (usize i = 0; i < n; ++i) y[i] = 0.0;
    if (n == 0 || h.empty()) return y;
    for (usize i = 0; i < n; ++i) {
        f64 acc = 0.0;
        for (usize j = 0; j < h.size(); ++j) {
            const usize shift = (j * step) % n;
            const usize idx = (i + n - shift) % n;
            acc += h[j] * x[idx];
        }
        y[i] = acc;
    }
    return y;
}

} // namespace

SwtResult swt_ad(ArrayRef<f64> x, i32 level, ArrayRef<f64> lo_d, ArrayRef<f64> hi_d, Allocator* alloc) {
    SwtResult res{Array<f64>(alloc), Array<f64>(alloc), 0, 0};
    if (level < 1 || lo_d.size() < 2 || (lo_d.size() & 1u) != 0u || lo_d.size() != hi_d.size()) return res;

    const usize n = x.size();
    const usize h_rows = static_cast<usize>(level);
    res.levels = h_rows;
    res.length = n;
    if (n == 0) return res;

    const usize needed = static_cast<usize>(1) << h_rows;
    if (n % needed != 0) return res;

    Array<f64> xv(alloc);
    xv.resize(n);
    for (usize i = 0; i < n; ++i) xv[i] = x[i];

    Array<f64> temp_lo(alloc);
    Array<f64> temp_hi(alloc);
    temp_lo.resize(lo_d.size());
    temp_hi.resize(hi_d.size());
    for (usize i = 0; i < lo_d.size(); ++i) {
        temp_lo[i] = lo_d[i];
        temp_hi[i] = hi_d[i];
    }

    res.swa_colmajor.resize(h_rows * n);
    res.swd_colmajor.resize(h_rows * n);

    auto conv_keep = [&](ArrayRef<f64> xe, ArrayRef<f64> h, usize lf) {
        Array<f64> out(alloc);
        out.resize(n);
        for (usize m = 0; m < n; ++m) out[m] = 0.0;
        for (usize j = 0; j < lf; ++j) {
            const f64 hj = h[j];
            if (hj == 0.0) continue;
            for (usize m = 0; m < n; ++m) {
                out[m] += hj * xe[lf + m - j];
            }
        }
        return out;
    };

    auto upsample_filter = [&](ArrayRef<f64> h) {
        Array<f64> u(alloc);
        u.resize(2 * h.size());
        for (usize t = 0; t < u.size(); ++t) u[t] = 0.0;
        for (usize t = 0; t < h.size(); ++t) u[2 * t] = h[t];
        return u;
    };

    for (usize k = 0; k < h_rows; ++k) {
        const usize lf = temp_lo.size();
        const usize ext = lf / 2;
        Array<f64> xe(alloc);
        xe.resize(n + 2 * ext);
        for (usize i = 0; i < ext; ++i) {
            xe[i] = xv[(n + i - (ext % n)) % n];
            xe[ext + n + i] = xv[i % n];
        }
        for (usize i = 0; i < n; ++i) xe[ext + i] = xv[i];

        Array<f64> d_row = conv_keep(ArrayRef<f64>(xe.data(), xe.size()), ArrayRef<f64>(temp_hi.data(), lf), lf);
        Array<f64> a_row = conv_keep(ArrayRef<f64>(xe.data(), xe.size()), ArrayRef<f64>(temp_lo.data(), lf), lf);
        for (usize c = 0; c < n; ++c) {
            res.swa_colmajor[c * h_rows + k] = a_row[c];
            res.swd_colmajor[c * h_rows + k] = d_row[c];
        }
        xv = move(a_row);
        if (k + 1 < h_rows) {
            temp_lo = upsample_filter(ArrayRef<f64>(temp_lo.data(), temp_lo.size()));
            temp_hi = upsample_filter(ArrayRef<f64>(temp_hi.data(), temp_hi.size()));
        }
    }
    return res;
}

Array<f64> modwt(ArrayRef<f64> x, i32 level, const FilterBank& fb, Allocator* alloc) {
    Array<f64> out(alloc);
    if (level < 1 || x.empty() || fb.lo_d.empty() || fb.hi_d.empty()) return out;

    const usize n = x.size();
    const usize lv = static_cast<usize>(level);
    const usize h_rows = lv + 1;
    const f64 inv_sqrt2 = 1.0 / sqrt(2.0);

    Array<f64> h_lo(alloc);
    Array<f64> h_hi(alloc);
    const usize lf = fb.lo_d.size();
    h_lo.resize(lf);
    h_hi.resize(lf);
    for (usize i = 0; i < lf; ++i) {
        h_lo[i] = fb.lo_d[lf - 1 - i] * inv_sqrt2;
        h_hi[i] = fb.hi_d[lf - 1 - i] * inv_sqrt2;
    }

    Array<f64> a(alloc);
    a.resize(n);
    for (usize i = 0; i < n; ++i) a[i] = x[i];

    out.resize(h_rows * n);
    for (usize k = 0; k < lv; ++k) {
        const usize step = static_cast<usize>(1) << k;
        Array<f64> a_next = circ_corr_back(ArrayRef<f64>(a.data(), n), ArrayRef<f64>(h_lo.data(), lf), step, alloc);
        Array<f64> d_next = circ_corr_back(ArrayRef<f64>(a.data(), n), ArrayRef<f64>(h_hi.data(), lf), step, alloc);
        for (usize c = 0; c < n; ++c) out[c * h_rows + k] = d_next[c];
        a = move(a_next);
    }
    for (usize c = 0; c < n; ++c) out[c * h_rows + lv] = a[c];
    return out;
}

Array<f64> imodwt(ArrayRef<f64> w_colmajor, usize levels, usize n, const FilterBank& fb, Allocator* alloc) {
    Array<f64> a(alloc);
    const usize h_rows = levels + 1;
    if (levels == 0 || n == 0 || w_colmajor.size() < h_rows * n || fb.lo_d.empty() || fb.hi_d.empty()) return a;

    const f64 inv_sqrt2 = 1.0 / sqrt(2.0);
    const usize lf = fb.lo_d.size();
    Array<f64> h_lo(alloc);
    Array<f64> h_hi(alloc);
    h_lo.resize(lf);
    h_hi.resize(lf);
    for (usize i = 0; i < lf; ++i) {
        h_lo[i] = fb.lo_d[lf - 1 - i] * inv_sqrt2;
        h_hi[i] = fb.hi_d[lf - 1 - i] * inv_sqrt2;
    }

    a.resize(n);
    for (usize c = 0; c < n; ++c) a[c] = w_colmajor[c * h_rows + levels];

    Array<f64> d(alloc);
    d.resize(n);
    for (usize kk = levels; kk > 0; --kk) {
        const usize k = kk - 1;
        const usize step = static_cast<usize>(1) << k;
        for (usize c = 0; c < n; ++c) d[c] = w_colmajor[c * h_rows + k];
        Array<f64> lo = circ_corr_fwd(ArrayRef<f64>(a.data(), n), ArrayRef<f64>(h_lo.data(), lf), step, alloc);
        Array<f64> hi = circ_corr_fwd(ArrayRef<f64>(d.data(), n), ArrayRef<f64>(h_hi.data(), lf), step, alloc);
        for (usize c = 0; c < n; ++c) a[c] = lo[c] + hi[c];
    }
    return a;
}

} // namespace keel::wavelet
