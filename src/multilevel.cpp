#include "keel/wavelet.h"

#include "keel/arena.h"
#include "keel/math.h"

namespace keel::wavelet {

namespace {

[[nodiscard]] bool mul_overflow(usize a, usize b, usize& out) noexcept {
    if (a != 0 && b > Limits<usize>::kMax / a) {
        return true;
    }
    out = a * b;
    return false;
}

Array<f64> conv_full_1d(ArrayRef<f64> a, ArrayRef<f64> b, Allocator* alloc) {
    Array<f64> c(alloc);
    if (a.empty() || b.empty()) {
        return c;
    }
    if (a.size() > Limits<usize>::kMax - (b.size() - 1)) {
        return c;
    }
    const usize out_n = a.size() + b.size() - 1;
    c.resize(out_n);
    for (usize i = 0; i < out_n; ++i) {
        c[i] = 0.0;
    }
    for (usize i = 0; i < a.size(); ++i) {
        const f64 ai = a[i];
        for (usize j = 0; j < b.size(); ++j) {
            c[i + j] += ai * b[j];
        }
    }
    return c;
}

} // namespace

bool wenergy(ArrayRef<f64> c, ArrayRef<usize> l, f64& ea_out, MutArrayRef<f64> ed_out) noexcept {
    if (l.size() < 3) {
        return false;
    }
    const usize n_lev = l.size() - 2;
    if (ed_out.size() < n_lev) {
        return false;
    }

    usize expected_c = l[0];
    if (expected_c > c.size()) {
        return false;
    }
    for (usize k = 0; k < n_lev; ++k) {
        const usize len = l[1 + k];
        if (len > c.size() - expected_c) {
            return false;
        }
        expected_c += len;
    }

    // Scale by max |c[i]| to prevent overflow/underflow in sum of squares.
    f64 max_abs = 0.0;
    for (usize i = 0; i < expected_c; ++i) {
        if (keel::is_nan(c[i])) {
            ea_out = kNaN;
            for (usize k = 0; k < n_lev; ++k) ed_out[k] = kNaN;
            return true;
        }
        const f64 a = keel::abs(c[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs == 0.0) {
        ea_out = 0.0;
        for (usize k = 0; k < n_lev; ++k) ed_out[k] = 0.0;
        return true;
    }

    const bool has_inf = keel::is_inf(max_abs);
    const f64 inv_max = has_inf ? 0.0 : (1.0 / max_abs);

    auto band_energy = [&](usize off, usize len) -> f64 {
        f64 s = 0.0;
        if (has_inf) {
            for (usize i = 0; i < len; ++i) {
                if (keel::is_inf(c[off + i])) s += 1.0;
            }
        } else {
            for (usize i = 0; i < len; ++i) {
                const f64 v = c[off + i] * inv_max;
                s += v * v;
            }
        }
        return s;
    };

    const f64 ea_raw = band_energy(0, l[0]);
    f64 total = ea_raw;
    usize off = l[0];
    for (usize k = 0; k < n_lev; ++k) {
        const usize len = l[1 + k];
        const f64 ek = band_energy(off, len);
        ed_out[n_lev - 1 - k] = ek;
        total += ek;
        off += len;
    }

    const f64 scale = (total > 0.0) ? (100.0 / total) : 0.0;
    ea_out = ea_raw * scale;
    for (usize k = 0; k < n_lev; ++k) {
        ed_out[k] *= scale;
    }
    return true;
}

Array<f64> upcoef(bool is_detail, ArrayRef<f64> x, const FilterBank& fb, i32 level, i64 target_len,
                  Allocator* alloc) {
    Array<f64> y(alloc);
    if (level < 0 || x.empty() || fb.lo_r.empty() || fb.hi_r.empty()) {
        return y;
    }
    if (level == 0) {
        y.resize(x.size());
        for (usize i = 0; i < x.size(); ++i) {
            y[i] = x[i];
        }
    } else {
        ArenaTemp scratch = keel::get_scratch(alloc);
        Array<f64> cur(scratch.allocator());
        cur.resize(x.size());
        for (usize i = 0; i < x.size(); ++i) {
            cur[i] = x[i];
        }
        for (i32 i = 0; i < level; ++i) {
            const ArrayRef<f64> filt =
                (i == 0 && is_detail) ? ArrayRef<f64>(fb.hi_r.data(), fb.hi_r.size())
                                      : ArrayRef<f64>(fb.lo_r.data(), fb.lo_r.size());
            if (cur.size() > Limits<usize>::kMax / 2) {
                return Array<f64>(alloc);
            }
            Array<f64> up(scratch.allocator());
            up.resize(2 * cur.size() - 1);
            for (usize k = 0; k < up.size(); ++k) {
                up[k] = 0.0;
            }
            for (usize k = 0; k < cur.size(); ++k) {
                up[2 * k] = cur[k];
            }
            Allocator* step_alloc = (i + 1 == level && target_len < 0) ? alloc : scratch.allocator();
            cur = conv_full_1d(ArrayRef<f64>(up.data(), up.size()), filt, step_alloc);
            if (cur.empty()) {
                return Array<f64>(alloc);
            }
        }
        if (target_len < 0) {
            return cur;
        }
        if (as<usize>(target_len) < cur.size()) {
            const usize keep = as<usize>(target_len);
            const usize start = (cur.size() - keep) / 2;
            y.resize(keep);
            for (usize i = 0; i < keep; ++i) {
                y[i] = cur[start + i];
            }
            return y;
        }
        y.resize(cur.size());
        for (usize i = 0; i < cur.size(); ++i) {
            y[i] = cur[i];
        }
        return y;
    }

    if (target_len >= 0 && as<usize>(target_len) < y.size()) {
        const usize keep = as<usize>(target_len);
        const usize start = (y.size() - keep) / 2;
        Array<f64> cropped(alloc);
        cropped.resize(keep);
        for (usize i = 0; i < keep; ++i) {
            cropped[i] = y[start + i];
        }
        return cropped;
    }
    return y;
}

Array<f64> wrcoef(bool is_detail, ArrayRef<f64> c, ArrayRef<usize> l, ArrayRef<f64> lo_r, ArrayRef<f64> hi_r,
                  i32 level, Allocator* alloc) {
    Array<f64> out(alloc);
    if (l.size() < 3 || l.size() > as<usize>(Limits<i32>::kMax) || lo_r.size() < 2 || hi_r.size() != lo_r.size()) {
        return out;
    }
    const i32 n_max = as<i32>(l.size()) - 2;
    const i32 eff_n = (level == -1) ? n_max : level;
    if (!is_detail) {
        if (eff_n < 0 || eff_n > n_max) {
            return out;
        }
    } else {
        if (eff_n < 1 || eff_n > n_max) {
            return out;
        }
    }

    usize expected_c = l[0];
    if (expected_c > c.size()) {
        return out;
    }
    for (i32 k = 0; k < n_max; ++k) {
        const usize step_len = l[1 + as<usize>(k)];
        if (step_len > c.size() - expected_c) {
            return out;
        }
        expected_c += step_len;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> cv(expected_c, scratch.allocator());
    for (usize i = 0; i < expected_c; ++i) {
        cv[i] = c[i];
    }

    Array<usize> d_start(as<usize>(n_max) + 1, 0, scratch.allocator());
    Array<usize> d_end(as<usize>(n_max) + 1, 0, scratch.allocator());
    {
        usize off = l[0];
        for (i32 idx_l = 1; idx_l <= n_max; ++idx_l) {
            const usize dlen = l[as<usize>(idx_l)];
            const usize lev = as<usize>(n_max - idx_l + 1);
            d_start[lev] = off;
            d_end[lev] = off + dlen;
            off += dlen;
        }
    }

    if (!is_detail) {
        for (i32 k = 1; k <= eff_n; ++k) {
            const usize ku = as<usize>(k);
            for (usize i = d_start[ku]; i < d_end[ku]; ++i) {
                cv[i] = 0.0;
            }
        }
    } else {
        for (usize i = 0; i < l[0]; ++i) {
            cv[i] = 0.0;
        }
        for (i32 k = 1; k <= n_max; ++k) {
            if (k == eff_n) {
                continue;
            }
            const usize ku = as<usize>(k);
            for (usize i = d_start[ku]; i < d_end[ku]; ++i) {
                cv[i] = 0.0;
            }
        }
    }

    return appcoef(ArrayRef<f64>(cv.data(), cv.size()), l, lo_r, hi_r, 0, alloc);
}

usize wmaxlev(usize signal_len, usize filter_len) noexcept {
    if (filter_len < 2 || signal_len < filter_len) {
        return 0;
    }
    usize q = signal_len / (filter_len - 1);
    usize lev = 0;
    while (q > 1) {
        q >>= 1;
        ++lev;
    }
    return lev;
}

Wavedec2Result wavedec2(ArrayRef<f64> x_colmajor, usize rows, usize cols, i32 level, const FilterBank& fb,
                        Allocator* alloc) {
    Wavedec2Result res{Array<f64>(alloc), Array<usize>(alloc), Array<usize>(alloc)};
    usize total_in = 0;
    if (level < 1 || rows == 0 || cols == 0 || mul_overflow(rows, cols, total_in) || x_colmajor.size() < total_in ||
        fb.lo_d.size() < 2 || fb.hi_d.size() != fb.lo_d.size()) {
        return res;
    }

    const usize lv = as<usize>(level);
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<Array<f64>> hs(scratch.allocator());
    Array<Array<f64>> vs(scratch.allocator());
    Array<Array<f64>> ds(scratch.allocator());
    Array<usize> det_r(lv, 0, scratch.allocator());
    Array<usize> det_c(lv, 0, scratch.allocator());
    hs.reserve(lv);
    vs.reserve(lv);
    ds.reserve(lv);

    Array<f64> running(total_in, scratch.allocator());
    for (usize i = 0; i < total_in; ++i) {
        running[i] = x_colmajor[i];
    }
    usize cur_r = rows;
    usize cur_c = cols;

    for (usize k = 0; k < lv; ++k) {
        Dwt2Result step = dwt2(ArrayRef<f64>(running.data(), running.size()), cur_r, cur_c, fb, scratch.allocator());
        if (step.out_rows == 0 || step.out_cols == 0) {
            return res;
        }
        det_r[k] = step.out_rows;
        det_c[k] = step.out_cols;
        hs.push_back(move(step.ch));
        vs.push_back(move(step.cv));
        ds.push_back(move(step.cd));
        running = move(step.ca);
        cur_r = step.out_rows;
        cur_c = step.out_cols;
    }

    usize total_c = running.size();
    for (usize k = 0; k < lv; ++k) {
        const usize band_sum = hs[k].size();
        if (band_sum > Limits<usize>::kMax / 3 || 3 * band_sum > Limits<usize>::kMax - total_c) {
            return res;
        }
        total_c += 3 * band_sum;
    }
    res.c.reserve(total_c);
    for (usize i = 0; i < running.size(); ++i) {
        res.c.push_back(running[i]);
    }
    for (usize k = lv; k > 0; --k) {
        const usize idx = k - 1;
        for (usize i = 0; i < hs[idx].size(); ++i) res.c.push_back(hs[idx][i]);
        for (usize i = 0; i < vs[idx].size(); ++i) res.c.push_back(vs[idx][i]);
        for (usize i = 0; i < ds[idx].size(); ++i) res.c.push_back(ds[idx][i]);
    }

    res.s_rows.reserve(lv + 2);
    res.s_cols.reserve(lv + 2);
    res.s_rows.push_back(cur_r);
    res.s_cols.push_back(cur_c);
    for (usize k = lv; k > 0; --k) {
        res.s_rows.push_back(det_r[k - 1]);
        res.s_cols.push_back(det_c[k - 1]);
    }
    res.s_rows.push_back(rows);
    res.s_cols.push_back(cols);
    return res;
}

Idwt2Result appcoef2(ArrayRef<f64> c, ArrayRef<usize> s_rows, ArrayRef<usize> s_cols, const FilterBank& fb,
                     i32 level, Allocator* alloc) {
    Idwt2Result res{Array<f64>(alloc), 0, 0};
    if (s_rows.size() < 3 || s_cols.size() != s_rows.size() || s_rows.size() > as<usize>(Limits<i32>::kMax)) {
        return res;
    }
    const i32 n_max = as<i32>(s_rows.size()) - 2;
    const i32 target = (level < 0) ? n_max : level;
    if (target < 0 || target > n_max) {
        return res;
    }
    if (target < n_max && (fb.lo_r.size() < 2 || fb.hi_r.size() != fb.lo_r.size())) {
        return res;
    }

    const usize ar = s_rows[0];
    const usize ac = s_cols[0];
    usize an = 0;
    if (ar == 0 || ac == 0 || mul_overflow(ar, ac, an) || an > c.size()) {
        return res;
    }
    if (target == n_max) {
        res.out.resize(an);
        for (usize i = 0; i < an; ++i) {
            res.out[i] = c[i];
        }
        res.rows = ar;
        res.cols = ac;
        return res;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> running(an, scratch.allocator());
    for (usize i = 0; i < an; ++i) {
        running[i] = c[i];
    }
    usize cur_r = ar;
    usize cur_c = ac;
    usize off = an;

    for (i32 lev = n_max; lev > target; --lev) {
        const usize s_idx = as<usize>(n_max - lev + 1);
        const usize dr = s_rows[s_idx];
        const usize dc = s_cols[s_idx];
        usize dn = 0;
        usize three_dn = 0;
        if (dr != cur_r || dc != cur_c || mul_overflow(dr, dc, dn) || mul_overflow(as<usize>(3), dn, three_dn) ||
            three_dn > c.size() - off) {
            return res;
        }
        const ArrayRef<f64> ch(c.data() + off, dn);
        const ArrayRef<f64> cv(c.data() + off + dn, dn);
        const ArrayRef<f64> cd(c.data() + off + 2 * dn, dn);
        off += three_dn;

        const i32 next_lev = lev - 1;
        const usize tgt_idx =
            (next_lev == 0) ? as<usize>(n_max + 1) : as<usize>(n_max - next_lev + 1);
        const usize tgt_r = s_rows[tgt_idx];
        const usize tgt_c = s_cols[tgt_idx];
        if (tgt_r > as<usize>(Limits<i64>::kMax) || tgt_c > as<usize>(Limits<i64>::kMax)) {
            return res;
        }

        Idwt2Result step = idwt2(ArrayRef<f64>(running.data(), running.size()), ch, cv, cd, cur_r, cur_c, fb,
                                 as<i64>(tgt_r), as<i64>(tgt_c),
                                 (next_lev == target) ? alloc : scratch.allocator());
        if (step.rows == 0 || step.cols == 0) {
            return res;
        }
        running = move(step.out);
        cur_r = step.rows;
        cur_c = step.cols;
    }

    res.out = move(running);
    res.rows = cur_r;
    res.cols = cur_c;
    return res;
}

Idwt2Result waverec2(ArrayRef<f64> c, ArrayRef<usize> s_rows, ArrayRef<usize> s_cols, const FilterBank& fb,
                     Allocator* alloc) {
    return appcoef2(c, s_rows, s_cols, fb, 0, alloc);
}

Idwt2Result detcoef2(Detail2D kind, ArrayRef<f64> c, ArrayRef<usize> s_rows, ArrayRef<usize> s_cols, i32 level,
                     Allocator* alloc) {
    Idwt2Result res{Array<f64>(alloc), 0, 0};
    const usize kind_idx = as<usize>(kind);
    if (kind_idx > 2 || s_rows.size() < 3 || s_cols.size() != s_rows.size() ||
        s_rows.size() > as<usize>(Limits<i32>::kMax)) {
        return res;
    }
    const i32 n_max = as<i32>(s_rows.size()) - 2;
    if (level < 1 || level > n_max) {
        return res;
    }

    usize off = 0;
    if (mul_overflow(s_rows[0], s_cols[0], off) || off > c.size()) {
        return res;
    }
    for (i32 lev = n_max; lev > level; --lev) {
        const usize idx = as<usize>(n_max - lev + 1);
        usize bn = 0;
        usize three_bn = 0;
        if (mul_overflow(s_rows[idx], s_cols[idx], bn) || mul_overflow(as<usize>(3), bn, three_bn) ||
            three_bn > c.size() - off) {
            return res;
        }
        off += three_bn;
    }

    const usize idx = as<usize>(n_max - level + 1);
    const usize dr = s_rows[idx];
    const usize dc = s_cols[idx];
    usize dn = 0;
    usize kind_off = 0;
    if (mul_overflow(dr, dc, dn) || mul_overflow(kind_idx, dn, kind_off) || kind_off > c.size() - off) {
        return res;
    }
    off += kind_off;
    if (dn > c.size() - off) {
        return res;
    }

    res.out.resize(dn);
    for (usize i = 0; i < dn; ++i) {
        res.out[i] = c[off + i];
    }
    res.rows = dr;
    res.cols = dc;
    return res;
}

i32 haar_max_level(usize n) noexcept {
    i32 k = 0;
    while (n > 1 && (n & 1u) == 0u) {
        ++k;
        n >>= 1;
    }
    return k;
}

void haar_step(ArrayRef<f64> x, MutArrayRef<f64> a_out, MutArrayRef<f64> d_out, bool integer_lifting) noexcept {
    const usize h = x.size() / 2;
    if (a_out.size() < h || d_out.size() < h) {
        return;
    }
    if (integer_lifting) {
        for (usize i = 0; i < h; ++i) {
            const f64 x0 = x[2 * i];
            const f64 x1 = x[2 * i + 1];
            const f64 dv = x1 - x0;
            d_out[i] = dv;
            a_out[i] = x0 + keel::floor(dv * 0.5);
        }
    } else {
        for (usize i = 0; i < h; ++i) {
            const f64 x0 = x[2 * i];
            const f64 x1 = x[2 * i + 1];
            a_out[i] = (x0 + x1) * kInvSqrt2;
            d_out[i] = (x1 - x0) * kInvSqrt2;
        }
    }
}

void haar_step_c64(ArrayRef<c64> x, MutArrayRef<c64> a_out, MutArrayRef<c64> d_out) noexcept {
    const usize h = x.size() / 2;
    if (a_out.size() < h || d_out.size() < h) {
        return;
    }
    for (usize i = 0; i < h; ++i) {
        const c64 x0 = x[2 * i];
        const c64 x1 = x[2 * i + 1];
        a_out[i] = (x0 + x1) * kInvSqrt2;
        d_out[i] = (x1 - x0) * kInvSqrt2;
    }
}

void ihaar_step(ArrayRef<f64> a, ArrayRef<f64> d, MutArrayRef<f64> x_out, bool integer_lifting) noexcept {
    const usize h = (a.size() < d.size()) ? a.size() : d.size();
    if (h > Limits<usize>::kMax / 2 || x_out.size() < 2 * h) {
        return;
    }
    if (integer_lifting) {
        for (usize i = 0; i < h; ++i) {
            const f64 ai = a[i];
            const f64 di = d[i];
            const f64 x0 = ai - keel::floor(di * 0.5);
            x_out[2 * i] = x0;
            x_out[2 * i + 1] = x0 + di;
        }
    } else {
        for (usize i = 0; i < h; ++i) {
            const f64 ai = a[i];
            const f64 di = d[i];
            x_out[2 * i] = (ai - di) * kInvSqrt2;
            x_out[2 * i + 1] = (ai + di) * kInvSqrt2;
        }
    }
}

void ihaar_step_c64(ArrayRef<c64> a, ArrayRef<f64> d, MutArrayRef<c64> x_out) noexcept {
    const usize h = (a.size() < d.size()) ? a.size() : d.size();
    if (h > Limits<usize>::kMax / 2 || x_out.size() < 2 * h) {
        return;
    }
    for (usize i = 0; i < h; ++i) {
        const c64 ai = a[i];
        const c64 di(d[i], 0.0);
        x_out[2 * i] = (ai - di) * kInvSqrt2;
        x_out[2 * i + 1] = (ai + di) * kInvSqrt2;
    }
}

void ihaar_step_c64(ArrayRef<c64> a, ArrayRef<c64> d, MutArrayRef<c64> x_out) noexcept {
    const usize h = (a.size() < d.size()) ? a.size() : d.size();
    if (h > Limits<usize>::kMax / 2 || x_out.size() < 2 * h) {
        return;
    }
    for (usize i = 0; i < h; ++i) {
        const c64 ai = a[i];
        const c64 di = d[i];
        x_out[2 * i] = (ai - di) * kInvSqrt2;
        x_out[2 * i + 1] = (ai + di) * kInvSqrt2;
    }
}

void dyaddown_1d(ArrayRef<f64> x, MutArrayRef<f64> out, bool keep_odd) noexcept {
    const usize n = x.size();
    const usize out_n = keep_odd ? (n / 2 + (n & 1u)) : (n / 2);
    if (out.size() < out_n) {
        return;
    }
    const usize start = keep_odd ? 0 : 1;
    for (usize k = 0; k < out_n; ++k) {
        out[k] = x[start + 2 * k];
    }
}

void dyadup_1d(ArrayRef<f64> x, MutArrayRef<f64> out, bool pad_ends) noexcept {
    const usize n = x.size();
    if (n == 0 || n > (Limits<usize>::kMax - 1) / 2) {
        return;
    }
    const usize out_n = pad_ends ? (2 * n + 1) : (2 * n - 1);
    if (out.size() < out_n) {
        return;
    }
    // Reverse iteration preserves x even when out aliases x in-place.
    if (pad_ends) {
        out[2 * n] = 0.0;
        for (usize i = n; i > 0; --i) {
            const usize k = i - 1;
            const f64 v = x[k];
            out[2 * k + 1] = v;
            out[2 * k] = 0.0;
        }
    } else {
        for (usize i = n; i > 0; --i) {
            const usize k = i - 1;
            const f64 v = x[k];
            out[2 * k] = v;
            if (k > 0) {
                out[2 * k - 1] = 0.0;
            }
        }
    }
}

} // namespace keel::wavelet
