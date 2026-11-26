#include "keel/wavelet.h"

#include "keel/arena.h"

namespace keel::wavelet {

namespace {

[[nodiscard]] bool mul_overflow(usize a, usize b, usize& out) noexcept {
    if (a != 0 && b > Limits<usize>::kMax / a) {
        return true;
    }
    out = a * b;
    return false;
}

Array<f64> sym_extend(ArrayRef<f64> x, usize pad, Allocator* alloc) {
    const usize n = x.size();
    Array<f64> y(alloc);
    if (n == 0 || pad > Limits<usize>::kMax / 2 || n > Limits<usize>::kMax - 2 * pad) {
        return y;
    }
    if (n > Limits<usize>::kMax / 2) {
        return y;
    }
    const usize period = 2 * n;
    y.resize(n + 2 * pad);
    for (usize i = 0; i < pad; ++i) {
        const usize rem = (i + 1) % period;
        const usize r = (rem == 0) ? 0 : (period - rem);
        y[pad - 1 - i] = (r < n) ? x[r] : x[period - 1 - r];
    }
    for (usize i = 0; i < n; ++i) {
        y[pad + i] = x[i];
    }
    for (usize i = 0; i < pad; ++i) {
        const usize r = (n + (i % period)) % period;
        y[pad + n + i] = (r < n) ? x[r] : x[period - 1 - r];
    }
    return y;
}

Array<f64> conv_full(ArrayRef<f64> a, ArrayRef<f64> b, Allocator* alloc) {
    Array<f64> y(alloc);
    if (a.empty() || b.empty()) return y;
    if (a.size() > Limits<usize>::kMax - (b.size() - 1)) return y;
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
    if (pad > Limits<usize>::kMax / 2 || n > Limits<usize>::kMax - 2 * pad) return res;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> ext = sym_extend(x, pad, scratch.allocator());
    if (ext.empty()) return res;
    const usize out_len = (n + lf - 1) / 2;

    auto conv_and_down = [&](ArrayRef<f64> h) {
        Array<f64> y = conv_full(ArrayRef<f64>(ext.data(), ext.size()), h, scratch.allocator());
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
    if (l == 0) return out;
    if (l > Limits<usize>::kMax / 2 || 2 * l > Limits<usize>::kMax - lf) return out;
    if (2 * l > as<usize>(Limits<i64>::kMax) || lf > as<usize>(Limits<i64>::kMax)) return out;

    ArenaTemp scratch = keel::get_scratch(alloc);
    auto upsample = [&](ArrayRef<f64> v) {
        Array<f64> u(scratch.allocator());
        u.resize(2 * l);
        for (usize i = 0; i < 2 * l; ++i) u[i] = 0.0;
        for (usize i = 0; i < v.size(); ++i) u[2 * i + 1] = v[i];
        return u;
    };

    Array<f64> up_a = upsample(ca);
    Array<f64> up_d = upsample(cd);
    Array<f64> y_a = conv_full(ArrayRef<f64>(up_a.data(), up_a.size()), lo_r, scratch.allocator());
    Array<f64> y_d = conv_full(ArrayRef<f64>(up_d.data(), up_d.size()), hi_r, scratch.allocator());

    const usize m = y_a.size();
    if (m < 2 || y_d.size() != m) return out;
    Array<f64> y(scratch.allocator());
    y.resize(m);
    for (usize i = 0; i < m; ++i) y[i] = y_a[i] + y_d[i];

    const usize sx = m - 1;
    const i64 want = (target_len >= 0) ? target_len : (as<i64>(2 * l) - as<i64>(lf) + 2);

    usize out_n = 0;
    usize offset = 1;
    if (want < 0 || as<usize>(want) >= sx) {
        out_n = sx;
        offset = 1;
    } else {
        out_n = as<usize>(want);
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
    if (level < 1 || x.empty() || fb.lo_d.size() < 2 || fb.hi_d.size() != fb.lo_d.size()) return res;

    const usize lv = as<usize>(level);
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<Array<f64>> details(scratch.allocator());
    details.reserve(lv);

    Array<f64> running(scratch.allocator());
    running.resize(x.size());
    for (usize i = 0; i < x.size(); ++i) running[i] = x[i];

    const ArrayRef<f64> lo_d(fb.lo_d.data(), fb.lo_d.size());
    const ArrayRef<f64> hi_d(fb.hi_d.data(), fb.hi_d.size());
    for (usize k = 0; k < lv; ++k) {
        Dwt1Result step = dwt(ArrayRef<f64>(running.data(), running.size()), lo_d, hi_d, scratch.allocator());
        if (step.ca.empty()) return res;
        details.push_back(move(step.cd));
        running = move(step.ca);
    }

    res.l.reserve(lv + 2);
    res.l.push_back(running.size());
    for (usize k = lv; k > 0; --k) res.l.push_back(details[k - 1].size());
    res.l.push_back(x.size());

    usize total = running.size();
    for (usize k = 0; k < lv; ++k) {
        if (details[k].size() > Limits<usize>::kMax - total) {
            return Wavedec1Result{Array<f64>(alloc), Array<usize>(alloc)};
        }
        total += details[k].size();
    }
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
    if (l.size() < 3 || l.size() > as<usize>(Limits<i32>::kMax)) return running;
    const i32 n_max = as<i32>(l.size()) - 2;
    const i32 target = (level < 0) ? n_max : level;
    if (target < 0 || target > n_max) return running;
    if (target < n_max && (lo_r.size() < 2 || hi_r.size() != lo_r.size())) return running;

    const usize a_len = l[0];
    if (a_len > c.size()) return running;
    if (target == n_max) {
        running.resize(a_len);
        for (usize i = 0; i < a_len; ++i) running[i] = c[i];
        return running;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> cur(scratch.allocator());
    cur.resize(a_len);
    for (usize i = 0; i < a_len; ++i) cur[i] = c[i];

    usize off = a_len;
    const usize steps = as<usize>(n_max - target);
    for (usize k = 0; k < steps; ++k) {
        const usize d_len = l[1 + k];
        if (d_len > c.size() - off) return Array<f64>(alloc);
        const usize rec_len = l[2 + k];
        if (rec_len > as<usize>(Limits<i64>::kMax)) return Array<f64>(alloc);
        Allocator* step_alloc = (k + 1 == steps) ? alloc : scratch.allocator();
        cur = idwt(ArrayRef<f64>(cur.data(), cur.size()), ArrayRef<f64>(c.data() + off, d_len), lo_r, hi_r,
                   as<i64>(rec_len), step_alloc);
        off += d_len;
    }
    return cur;
}

Array<f64> waverec(ArrayRef<f64> c, ArrayRef<usize> l, const FilterBank& fb, Allocator* alloc) {
    return appcoef(c, l, ArrayRef<f64>(fb.lo_r.data(), fb.lo_r.size()), ArrayRef<f64>(fb.hi_r.data(), fb.hi_r.size()),
                   0, alloc);
}

Array<f64> detcoef(ArrayRef<f64> c, ArrayRef<usize> l, i32 level, Allocator* alloc) {
    Array<f64> out(alloc);
    if (l.size() < 3 || l.size() > as<usize>(Limits<i32>::kMax)) return out;
    const i32 n_max = as<i32>(l.size()) - 2;
    if (level < 1 || level > n_max) return out;

    usize off = l[0];
    if (off > c.size()) return out;
    for (i32 k = 0; k < n_max - level; ++k) {
        const usize step_len = l[1 + as<usize>(k)];
        if (step_len > c.size() - off) return out;
        off += step_len;
    }
    const usize d_len = l[as<usize>(n_max - level + 1)];
    if (d_len > c.size() - off) return out;
    out.resize(d_len);
    for (usize i = 0; i < d_len; ++i) out[i] = c[off + i];
    return out;
}

// ============================================================================
// 2D Separable DWT / IDWT
// ============================================================================

Dwt2Result dwt2(ArrayRef<f64> x_colmajor, usize rows, usize cols, const FilterBank& fb, Allocator* alloc) {
    Dwt2Result res{Array<f64>(alloc), Array<f64>(alloc), Array<f64>(alloc), Array<f64>(alloc), 0, 0};
    usize total_in = 0;
    if (rows == 0 || cols == 0 || mul_overflow(rows, cols, total_in) || x_colmajor.size() < total_in) return res;

    const ArrayRef<f64> lo_d(fb.lo_d.data(), fb.lo_d.size());
    const ArrayRef<f64> hi_d(fb.hi_d.data(), fb.hi_d.size());
    const usize lf = lo_d.size();
    if (lf < 2 || hi_d.size() != lf) return res;
    if (cols > Limits<usize>::kMax - (lf - 1) || rows > Limits<usize>::kMax - (lf - 1)) return res;

    const usize out_cols = (cols + lf - 1) / 2;
    const usize out_rows = (rows + lf - 1) / 2;
    usize inter_n = 0;
    usize band_n = 0;
    if (mul_overflow(rows, out_cols, inter_n) || mul_overflow(out_rows, out_cols, band_n)) return res;

    ArenaTemp scratch = keel::get_scratch(alloc);

    // Pass 1: 1D DWT along each row -> lo_r_mat, hi_r_mat (rows x out_cols, col-major).
    Array<f64> lo_r_mat(scratch.allocator());
    Array<f64> hi_r_mat(scratch.allocator());
    lo_r_mat.resize(inter_n);
    hi_r_mat.resize(inter_n);

    Array<f64> row_buf(scratch.allocator());
    row_buf.resize(cols);
    for (usize r = 0; r < rows; ++r) {
        for (usize c = 0; c < cols; ++c) row_buf[c] = x_colmajor[c * rows + r];
        Dwt1Result dr = dwt(ArrayRef<f64>(row_buf.data(), cols), lo_d, hi_d, scratch.allocator());
        if (dr.ca.size() != out_cols || dr.cd.size() != out_cols) return res;
        for (usize c = 0; c < out_cols; ++c) {
            lo_r_mat[c * rows + r] = dr.ca[c];
            hi_r_mat[c * rows + r] = dr.cd[c];
        }
    }

    // Pass 2: 1D DWT along each column of lo_r_mat (-> ca, ch) and hi_r_mat (-> cv, cd).
    res.ca.resize(band_n);
    res.ch.resize(band_n);
    res.cv.resize(band_n);
    res.cd.resize(band_n);

    for (usize c = 0; c < out_cols; ++c) {
        Dwt1Result d_lo = dwt(ArrayRef<f64>(lo_r_mat.data() + c * rows, rows), lo_d, hi_d, scratch.allocator());
        Dwt1Result d_hi = dwt(ArrayRef<f64>(hi_r_mat.data() + c * rows, rows), lo_d, hi_d, scratch.allocator());
        if (d_lo.ca.size() != out_rows || d_hi.ca.size() != out_rows) {
            return Dwt2Result{Array<f64>(alloc), Array<f64>(alloc), Array<f64>(alloc), Array<f64>(alloc), 0, 0};
        }
        for (usize r = 0; r < out_rows; ++r) {
            res.ca[c * out_rows + r] = d_lo.ca[r];
            res.ch[c * out_rows + r] = d_lo.cd[r];
            res.cv[c * out_rows + r] = d_hi.ca[r];
            res.cd[c * out_rows + r] = d_hi.cd[r];
        }
    }
    res.out_rows = out_rows;
    res.out_cols = out_cols;
    return res;
}

Idwt2Result idwt2(ArrayRef<f64> ca, ArrayRef<f64> ch, ArrayRef<f64> cv, ArrayRef<f64> cd, usize rows, usize cols,
                  const FilterBank& fb, i64 target_rows, i64 target_cols, Allocator* alloc) {
    Idwt2Result res{Array<f64>(alloc), 0, 0};
    usize n_band = 0;
    if (rows == 0 || cols == 0 || mul_overflow(rows, cols, n_band) || ca.size() < n_band || ch.size() < n_band ||
        cv.size() < n_band || cd.size() < n_band) {
        return res;
    }

    const ArrayRef<f64> lo_r(fb.lo_r.data(), fb.lo_r.size());
    const ArrayRef<f64> hi_r(fb.hi_r.data(), fb.hi_r.size());
    const usize lf = lo_r.size();
    if (lf < 2 || hi_r.size() != lf) return res;
    if (rows > Limits<usize>::kMax / 2 || cols > Limits<usize>::kMax / 2 || 2 * rows > Limits<usize>::kMax - lf ||
        2 * cols > Limits<usize>::kMax - lf) {
        return res;
    }
    if (target_rows == 0 || target_cols == 0) {
        return res;
    }

    auto resolved_len = [lf](usize la, i64 tgt) -> usize {
        const usize sx = 2 * la + lf - 2;
        const i64 want = (tgt >= 0) ? tgt : (as<i64>(2 * la) - as<i64>(lf) + 2);
        if (want < 0 || as<usize>(want) >= sx) return sx;
        return as<usize>(want);
    };

    const usize out_rows = resolved_len(rows, target_rows);
    const usize out_cols = resolved_len(cols, target_cols);
    usize inter_n = 0;
    usize out_n = 0;
    if (out_rows == 0 || out_cols == 0 || mul_overflow(out_rows, cols, inter_n) ||
        mul_overflow(out_rows, out_cols, out_n)) {
        return res;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);

    // Pass 1: 1D idwt along columns -> lo_r_mat, hi_r_mat (out_rows x cols).
    Array<f64> lo_r_mat(scratch.allocator());
    Array<f64> hi_r_mat(scratch.allocator());
    lo_r_mat.resize(inter_n);
    hi_r_mat.resize(inter_n);

    for (usize c = 0; c < cols; ++c) {
        Array<f64> col_lo = idwt(ArrayRef<f64>(ca.data() + c * rows, rows), ArrayRef<f64>(ch.data() + c * rows, rows),
                                 lo_r, hi_r, as<i64>(out_rows), scratch.allocator());
        Array<f64> col_hi = idwt(ArrayRef<f64>(cv.data() + c * rows, rows), ArrayRef<f64>(cd.data() + c * rows, rows),
                                 lo_r, hi_r, as<i64>(out_rows), scratch.allocator());
        if (col_lo.size() != out_rows || col_hi.size() != out_rows) return res;
        for (usize r = 0; r < out_rows; ++r) {
            lo_r_mat[c * out_rows + r] = col_lo[r];
            hi_r_mat[c * out_rows + r] = col_hi[r];
        }
    }

    // Pass 2: 1D idwt along rows -> res.out (out_rows x out_cols).
    res.out.resize(out_n);
    Array<f64> row_a(scratch.allocator());
    Array<f64> row_d(scratch.allocator());
    row_a.resize(cols);
    row_d.resize(cols);

    for (usize r = 0; r < out_rows; ++r) {
        for (usize c = 0; c < cols; ++c) {
            row_a[c] = lo_r_mat[c * out_rows + r];
            row_d[c] = hi_r_mat[c * out_rows + r];
        }
        Array<f64> row_y = idwt(ArrayRef<f64>(row_a.data(), cols), ArrayRef<f64>(row_d.data(), cols), lo_r, hi_r,
                                as<i64>(out_cols), scratch.allocator());
        if (row_y.size() != out_cols) {
            return Idwt2Result{Array<f64>(alloc), 0, 0};
        }
        for (usize c = 0; c < out_cols; ++c) {
            res.out[c * out_rows + r] = row_y[c];
        }
    }

    res.rows = out_rows;
    res.cols = out_cols;
    return res;
}

} // namespace keel::wavelet
