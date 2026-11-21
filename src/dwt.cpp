#include "keel/wavelet.h"

namespace keel::wavelet {

namespace {

Array<f64> sym_extend(ArrayRef<f64> x, usize pad, Allocator* alloc) {
    const usize n = x.size();
    Array<f64> y(alloc);
    y.resize(n + 2 * pad);
    for (usize i = 0; i < pad; ++i) {
        y[pad - 1 - i] = (i < n) ? x[i] : x[n - 1];
    }
    for (usize i = 0; i < n; ++i) y[pad + i] = x[i];
    for (usize i = 0; i < pad; ++i) {
        y[pad + n + i] = (i + 1 <= n) ? x[n - 1 - i] : x[0];
    }
    return y;
}

Array<f64> conv_full(ArrayRef<f64> a, ArrayRef<f64> b, Allocator* alloc) {
    Array<f64> y(alloc);
    if (a.empty() || b.empty()) return y;
    const usize out_n = a.size() + b.size() - 1;
    y.resize(out_n);
    for (usize i = 0; i < out_n; ++i) y[i] = 0.0;
    for (usize i = 0; i < a.size(); ++i) {
        const f64 ai = a[i];
        for (usize j = 0; j < b.size(); ++j) {
            y[i + j] += ai * b[j];
        }
    }
    return y;
}

} // namespace

Dwt1Result dwt(ArrayRef<f64> x, ArrayRef<f64> lo_d, ArrayRef<f64> hi_d, Allocator* alloc) {
    Dwt1Result res{Array<f64>(alloc), Array<f64>(alloc)};
    const usize lf = lo_d.size();
    const usize n = x.size();
    if (lf < 2 || hi_d.size() != lf || n == 0) return res;

    const usize pad = lf - 1;
    Array<f64> ext = sym_extend(x, pad, alloc);
    const usize out_len = (n + lf - 1) / 2;

    auto conv_and_down = [&](ArrayRef<f64> h) {
        Array<f64> y = conv_full(ArrayRef<f64>(ext.data(), ext.size()), h, alloc);
        Array<f64> z(alloc);
        z.resize(out_len);
        for (usize k = 0; k < out_len; ++k) {
            const usize idx = (lf - 1) + 2 * k + 1;
            z[k] = (idx < y.size()) ? y[idx] : 0.0;
        }
        return z;
    };

    res.ca = conv_and_down(lo_d);
    res.cd = conv_and_down(hi_d);
    return res;
}

Array<f64> idwt(ArrayRef<f64> ca, ArrayRef<f64> cd, ArrayRef<f64> lo_r, ArrayRef<f64> hi_r, i64 target_len,
                Allocator* alloc) {
    Array<f64> out(alloc);
    const usize lf = lo_r.size();
    if (lf < 2 || hi_r.size() != lf) return out;

    const usize la = ca.size();
    const usize ld = cd.size();
    const usize l = max(la, ld);

    auto upsample = [&](ArrayRef<f64> v) {
        Array<f64> u(alloc);
        u.resize(2 * l);
        for (usize i = 0; i < 2 * l; ++i) u[i] = 0.0;
        for (usize i = 0; i < v.size(); ++i) u[2 * i + 1] = v[i];
        return u;
    };

    Array<f64> up_a = upsample(ca);
    Array<f64> up_d = upsample(cd);
    Array<f64> y_a = conv_full(ArrayRef<f64>(up_a.data(), up_a.size()), lo_r, alloc);
    Array<f64> y_d = conv_full(ArrayRef<f64>(up_d.data(), up_d.size()), hi_r, alloc);

    const usize m = y_a.size();
    if (m == 0) return out;
    Array<f64> y(alloc);
    y.resize(m);
    for (usize i = 0; i < m; ++i) y[i] = y_a[i] + y_d[i];

    const usize sx = m - 1;
    const i64 want = (target_len >= 0) ? target_len : (static_cast<i64>(2 * la) - static_cast<i64>(lf) + 2);

    usize out_n = 0;
    usize offset = 1;
    if (want < 0 || static_cast<usize>(want) >= sx) {
        out_n = sx;
        offset = 1;
    } else {
        out_n = static_cast<usize>(want);
        offset = (sx - out_n) / 2 + 1;
    }

    out.resize(out_n);
    for (usize i = 0; i < out_n; ++i) {
        const usize k = offset + i;
        out[i] = (k < m) ? y[k] : 0.0;
    }
    return out;
}

Wavedec1Result wavedec(ArrayRef<f64> x, i32 level, const FilterBank& fb, Allocator* alloc) {
    Wavedec1Result res{Array<f64>(alloc), Array<usize>(alloc)};
    if (level < 1) return res;

    const usize lv = static_cast<usize>(level);
    Array<Array<f64>> details(alloc);
    details.reserve(lv);

    Array<f64> running(alloc);
    running.resize(x.size());
    for (usize i = 0; i < x.size(); ++i) running[i] = x[i];

    for (usize k = 0; k < lv; ++k) {
        Dwt1Result step = dwt(ArrayRef<f64>(running.data(), running.size()), ArrayRef<f64>(fb.lo_d.data(), fb.lo_d.size()),
                              ArrayRef<f64>(fb.hi_d.data(), fb.hi_d.size()), alloc);
        details.push_back(move(step.cd));
        running = move(step.ca);
    }

    res.l.reserve(lv + 2);
    res.l.push_back(running.size());
    for (usize k = lv; k > 0; --k) res.l.push_back(details[k - 1].size());
    res.l.push_back(x.size());

    usize total = running.size();
    for (usize k = 0; k < lv; ++k) total += details[k].size();
    res.c.reserve(total);
    for (usize i = 0; i < running.size(); ++i) res.c.push_back(running[i]);
    for (usize k = lv; k > 0; --k) {
        const Array<f64>& dk = details[k - 1];
        for (usize i = 0; i < dk.size(); ++i) res.c.push_back(dk[i]);
    }
    return res;
}

Array<f64> appcoef(ArrayRef<f64> c, ArrayRef<usize> l, ArrayRef<f64> lo_r, ArrayRef<f64> hi_r, i32 level,
                   Allocator* alloc) {
    Array<f64> running(alloc);
    if (l.size() < 3) return running;
    const i32 n_max = static_cast<i32>(l.size()) - 2;
    const i32 target = (level < 0) ? n_max : level;
    if (target < 0 || target > n_max) return running;

    const usize a_len = l[0];
    if (a_len > c.size()) return running;
    running.resize(a_len);
    for (usize i = 0; i < a_len; ++i) running[i] = c[i];
    if (target == n_max) return running;

    usize off = a_len;
    const usize steps = static_cast<usize>(n_max - target);
    for (usize k = 0; k < steps; ++k) {
        const usize d_len = l[1 + k];
        if (off + d_len > c.size()) return Array<f64>(alloc);
        const usize rec_len = l[2 + k];
        running = idwt(ArrayRef<f64>(running.data(), running.size()), ArrayRef<f64>(c.data() + off, d_len), lo_r, hi_r,
                       static_cast<i64>(rec_len), alloc);
        off += d_len;
    }
    return running;
}

Array<f64> waverec(ArrayRef<f64> c, ArrayRef<usize> l, const FilterBank& fb, Allocator* alloc) {
    return appcoef(c, l, ArrayRef<f64>(fb.lo_r.data(), fb.lo_r.size()), ArrayRef<f64>(fb.hi_r.data(), fb.hi_r.size()),
                   0, alloc);
}

Array<f64> detcoef(ArrayRef<f64> c, ArrayRef<usize> l, i32 level, Allocator* alloc) {
    Array<f64> out(alloc);
    if (l.size() < 3) return out;
    const i32 n_max = static_cast<i32>(l.size()) - 2;
    if (level < 1 || level > n_max) return out;

    usize off = l[0];
    for (i32 k = 0; k < n_max - level; ++k) off += l[1 + static_cast<usize>(k)];
    const usize d_len = l[static_cast<usize>(n_max - level + 1)];
    if (off + d_len > c.size()) return out;
    out.resize(d_len);
    for (usize i = 0; i < d_len; ++i) out[i] = c[off + i];
    return out;
}

// ============================================================================
// 2D Separable DWT / IDWT
// ============================================================================

Dwt2Result dwt2(ArrayRef<f64> x_colmajor, usize rows, usize cols, const FilterBank& fb, Allocator* alloc) {
    Dwt2Result res{Array<f64>(alloc), Array<f64>(alloc), Array<f64>(alloc), Array<f64>(alloc), 0, 0};
    if (rows == 0 || cols == 0 || x_colmajor.size() < rows * cols) return res;

    ArrayRef<f64> lo_d(fb.lo_d.data(), fb.lo_d.size());
    ArrayRef<f64> hi_d(fb.hi_d.data(), fb.hi_d.size());
    const usize lf = lo_d.size();
    if (lf < 2 || hi_d.size() != lf) return res;

    const usize out_cols = (cols + lf - 1) / 2;
    const usize out_rows = (rows + lf - 1) / 2;
    res.out_rows = out_rows;
    res.out_cols = out_cols;

    // Pass 1: 1D DWT along each row -> lo_r_mat, hi_r_mat (rows x out_cols, col-major).
    Array<f64> lo_r_mat(alloc);
    Array<f64> hi_r_mat(alloc);
    lo_r_mat.resize(rows * out_cols);
    hi_r_mat.resize(rows * out_cols);

    Array<f64> row_buf(alloc);
    row_buf.resize(cols);
    for (usize r = 0; r < rows; ++r) {
        for (usize c = 0; c < cols; ++c) row_buf[c] = x_colmajor[c * rows + r];
        Dwt1Result dr = dwt(ArrayRef<f64>(row_buf.data(), cols), lo_d, hi_d, alloc);
        for (usize c = 0; c < out_cols; ++c) {
            lo_r_mat[c * rows + r] = dr.ca[c];
            hi_r_mat[c * rows + r] = dr.cd[c];
        }
    }

    // Pass 2: 1D DWT along each column of lo_r_mat (-> ca, ch) and hi_r_mat (-> cv, cd).
    const usize band_n = out_rows * out_cols;
    res.ca.resize(band_n);
    res.ch.resize(band_n);
    res.cv.resize(band_n);
    res.cd.resize(band_n);

    for (usize c = 0; c < out_cols; ++c) {
        Dwt1Result d_lo = dwt(ArrayRef<f64>(lo_r_mat.data() + c * rows, rows), lo_d, hi_d, alloc);
        Dwt1Result d_hi = dwt(ArrayRef<f64>(hi_r_mat.data() + c * rows, rows), lo_d, hi_d, alloc);
        for (usize r = 0; r < out_rows; ++r) {
            res.ca[c * out_rows + r] = d_lo.ca[r];
            res.ch[c * out_rows + r] = d_lo.cd[r];
            res.cv[c * out_rows + r] = d_hi.ca[r];
            res.cd[c * out_rows + r] = d_hi.cd[r];
        }
    }
    return res;
}

Idwt2Result idwt2(ArrayRef<f64> ca, ArrayRef<f64> ch, ArrayRef<f64> cv, ArrayRef<f64> cd, usize rows, usize cols,
                  const FilterBank& fb, i64 target_rows, i64 target_cols, Allocator* alloc) {
    Idwt2Result res{Array<f64>(alloc), 0, 0};
    const usize n_band = rows * cols;
    if (rows == 0 || cols == 0 || ca.size() < n_band || ch.size() < n_band || cv.size() < n_band ||
        cd.size() < n_band) {
        return res;
    }

    ArrayRef<f64> lo_r(fb.lo_r.data(), fb.lo_r.size());
    ArrayRef<f64> hi_r(fb.hi_r.data(), fb.hi_r.size());
    const usize lf = lo_r.size();
    if (lf < 2 || hi_r.size() != lf) return res;

    // Natural reconstruction sizes via 1D idwt rule.
    auto natural_len = [lf](usize la) -> usize {
        const usize sx = 2 * la + lf - 2;
        const i64 want = static_cast<i64>(2 * la) - static_cast<i64>(lf) + 2;
        if (want < 0 || static_cast<usize>(want) >= sx) return sx;
        return static_cast<usize>(want);
    };

    const usize nat_rows = natural_len(rows);
    const usize nat_cols = natural_len(cols);

    // Pass 1: 1D idwt along columns -> lo_r_mat, hi_r_mat (nat_rows x cols).
    Array<f64> lo_r_mat(alloc);
    Array<f64> hi_r_mat(alloc);
    lo_r_mat.resize(nat_rows * cols);
    hi_r_mat.resize(nat_rows * cols);

    for (usize c = 0; c < cols; ++c) {
        Array<f64> col_lo = idwt(ArrayRef<f64>(ca.data() + c * rows, rows), ArrayRef<f64>(ch.data() + c * rows, rows),
                                 lo_r, hi_r, static_cast<i64>(nat_rows), alloc);
        Array<f64> col_hi = idwt(ArrayRef<f64>(cv.data() + c * rows, rows), ArrayRef<f64>(cd.data() + c * rows, rows),
                                 lo_r, hi_r, static_cast<i64>(nat_rows), alloc);
        for (usize r = 0; r < nat_rows; ++r) {
            lo_r_mat[c * nat_rows + r] = col_lo[r];
            hi_r_mat[c * nat_rows + r] = col_hi[r];
        }
    }

    // Pass 2: 1D idwt along rows -> full_out (nat_rows x nat_cols).
    Array<f64> full_out(alloc);
    full_out.resize(nat_rows * nat_cols);
    Array<f64> row_a(alloc);
    Array<f64> row_d(alloc);
    row_a.resize(cols);
    row_d.resize(cols);

    for (usize r = 0; r < nat_rows; ++r) {
        for (usize c = 0; c < cols; ++c) {
            row_a[c] = lo_r_mat[c * nat_rows + r];
            row_d[c] = hi_r_mat[c * nat_rows + r];
        }
        Array<f64> row_y = idwt(ArrayRef<f64>(row_a.data(), cols), ArrayRef<f64>(row_d.data(), cols), lo_r, hi_r,
                                static_cast<i64>(nat_cols), alloc);
        for (usize c = 0; c < nat_cols; ++c) {
            full_out[c * nat_rows + r] = row_y[c];
        }
    }

    const usize keep_r = (target_rows >= 0) ? min(static_cast<usize>(target_rows), nat_rows) : nat_rows;
    const usize keep_c = (target_cols >= 0) ? min(static_cast<usize>(target_cols), nat_cols) : nat_cols;

    if (keep_r == nat_rows && keep_c == nat_cols) {
        res.out = move(full_out);
        res.rows = nat_rows;
        res.cols = nat_cols;
        return res;
    }

    const usize r0 = (nat_rows - keep_r) / 2;
    const usize c0 = (nat_cols - keep_c) / 2;
    res.out.resize(keep_r * keep_c);
    for (usize c = 0; c < keep_c; ++c) {
        for (usize r = 0; r < keep_r; ++r) {
            res.out[c * keep_r + r] = full_out[(c0 + c) * nat_rows + (r0 + r)];
        }
    }
    res.rows = keep_r;
    res.cols = keep_c;
    return res;
}

} // namespace keel::wavelet
