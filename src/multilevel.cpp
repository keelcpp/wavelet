#include "keel/wavelet.h"

#include "keel/arena.h"
#include "keel/math.h"

namespace keel::wavelet {

namespace {

Array<f64> conv_full_1d(ArrayRef<f64> a, ArrayRef<f64> b, Allocator* alloc) {
    Array<f64> c(alloc);
    if (a.empty() || b.empty()) {
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

    auto band_energy = [&](usize off, usize len) -> f64 {
        f64 s = 0.0;
        for (usize i = off; i < off + len && i < c.size(); ++i) {
            s += c[i] * c[i];
        }
        return s;
    };

    f64 total = 0.0;
    for (usize i = 0; i < c.size(); ++i) {
        total += c[i] * c[i];
    }
    const f64 scale = (total > 0.0) ? (100.0 / total) : 0.0;

    const usize a_len = l[0];
    ea_out = band_energy(0, a_len) * scale;

    usize off = a_len;
    for (usize k = 0; k < n_lev; ++k) {
        const usize len = l[1 + k];
        ed_out[n_lev - 1 - k] = band_energy(off, len) * scale;
        off += len;
    }
    return true;
}

Array<f64> upcoef(bool is_detail, ArrayRef<f64> x, const FilterBank& fb, i32 level, i64 target_len,
                  Allocator* alloc) {
    Array<f64> y(alloc);
    if (level < 0 || fb.lo_r.empty() || fb.hi_r.empty()) {
        return y;
    }
    y.resize(x.size());
    for (usize i = 0; i < x.size(); ++i) {
        y[i] = x[i];
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    for (i32 i = 0; i < level; ++i) {
        const ArrayRef<f64> filt =
            (i == 0 && is_detail) ? ArrayRef<f64>(fb.hi_r.data(), fb.hi_r.size())
                                  : ArrayRef<f64>(fb.lo_r.data(), fb.lo_r.size());
        Array<f64> up(scratch.allocator());
        if (!y.empty()) {
            up.resize(2 * y.size() - 1);
            for (usize k = 0; k < up.size(); ++k) {
                up[k] = 0.0;
            }
            for (usize k = 0; k < y.size(); ++k) {
                up[2 * k] = y[k];
            }
        }
        y = conv_full_1d(ArrayRef<f64>(up.data(), up.size()), filt, alloc);
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
    if (l.size() < 3 || lo_r.size() < 2 || hi_r.size() != lo_r.size()) {
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
    for (i32 k = 0; k < n_max; ++k) {
        expected_c += l[1 + as<usize>(k)];
    }
    if (c.size() < expected_c) {
        return out;
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
    const f64 r = as<f64>(signal_len) / as<f64>(filter_len - 1);
    const f64 lev = keel::floor(keel::log2(r));
    return (lev > 0.0) ? as<usize>(lev) : 0;
}

Wavedec2Result wavedec2(ArrayRef<f64> x_colmajor, usize rows, usize cols, i32 level, const FilterBank& fb,
                        Allocator* alloc) {
    Wavedec2Result res{Array<f64>(alloc), Array<usize>(alloc), Array<usize>(alloc)};
    if (level < 1 || rows == 0 || cols == 0 || x_colmajor.size() < rows * cols) {
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

    Array<f64> running(rows * cols, scratch.allocator());
    for (usize i = 0; i < rows * cols; ++i) {
        running[i] = x_colmajor[i];
    }
    usize cur_r = rows;
    usize cur_c = cols;

    for (usize k = 0; k < lv; ++k) {
        Dwt2Result step = dwt2(ArrayRef<f64>(running.data(), running.size()), cur_r, cur_c, fb, scratch.allocator());
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
        total_c += hs[k].size() + vs[k].size() + ds[k].size();
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
    if (s_rows.size() < 3 || s_cols.size() != s_rows.size()) {
        return res;
    }
    const i32 n_max = as<i32>(s_rows.size()) - 2;
    const i32 target = (level < 0) ? n_max : level;
    if (target < 0 || target > n_max) {
        return res;
    }

    const usize ar = s_rows[0];
    const usize ac = s_cols[0];
    const usize an = ar * ac;
    Array<f64> running(alloc);
    running.resize(an);
    for (usize i = 0; i < an; ++i) {
        running[i] = (i < c.size()) ? c[i] : 0.0;
    }
    if (target == n_max) {
        res.out = move(running);
        res.rows = ar;
        res.cols = ac;
        return res;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    usize cur_r = ar;
    usize cur_c = ac;
    usize off = an;

    auto get_slice = [&](usize start, usize len, Array<f64>& dst) {
        dst.resize(len);
        for (usize i = 0; i < len; ++i) {
            dst[i] = (start + i < c.size()) ? c[start + i] : 0.0;
        }
    };

    Array<f64> ch(scratch.allocator());
    Array<f64> cv(scratch.allocator());
    Array<f64> cd(scratch.allocator());

    for (i32 lev = n_max; lev > target; --lev) {
        const usize s_idx = as<usize>(n_max - lev + 1);
        const usize dr = s_rows[s_idx];
        const usize dc = s_cols[s_idx];
        const usize dn = dr * dc;
        get_slice(off, dn, ch);
        off += dn;
        get_slice(off, dn, cv);
        off += dn;
        get_slice(off, dn, cd);
        off += dn;

        const i32 next_lev = lev - 1;
        const usize tgt_idx =
            (next_lev == 0) ? as<usize>(n_max + 1) : as<usize>(n_max - next_lev + 1);
        const usize tgt_r = s_rows[tgt_idx];
        const usize tgt_c = s_cols[tgt_idx];

        Idwt2Result step = idwt2(ArrayRef<f64>(running.data(), running.size()), ArrayRef<f64>(ch.data(), dn),
                                 ArrayRef<f64>(cv.data(), dn), ArrayRef<f64>(cd.data(), dn), cur_r, cur_c, fb,
                                 as<i64>(tgt_r), as<i64>(tgt_c),
                                 (next_lev == target) ? alloc : scratch.allocator());
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
    if (s_rows.size() < 3 || s_cols.size() != s_rows.size()) {
        return res;
    }
    const i32 n_max = as<i32>(s_rows.size()) - 2;
    if (level < 1 || level > n_max) {
        return res;
    }

    usize off = s_rows[0] * s_cols[0];
    for (i32 lev = n_max; lev > level; --lev) {
        const usize idx = as<usize>(n_max - lev + 1);
        off += 3 * s_rows[idx] * s_cols[idx];
    }

    const usize idx = as<usize>(n_max - level + 1);
    const usize dr = s_rows[idx];
    const usize dc = s_cols[idx];
    const usize dn = dr * dc;
    off += as<usize>(kind) * dn;
    if (off + dn > c.size()) {
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
            const f64 dv = x[2 * i + 1] - x[2 * i];
            d_out[i] = dv;
            a_out[i] = x[2 * i] + keel::floor(dv * 0.5);
        }
    } else {
        for (usize i = 0; i < h; ++i) {
            a_out[i] = (x[2 * i] + x[2 * i + 1]) * kInvSqrt2;
            d_out[i] = (x[2 * i + 1] - x[2 * i]) * kInvSqrt2;
        }
    }
}

void haar_step_c64(ArrayRef<c64> x, MutArrayRef<c64> a_out, MutArrayRef<c64> d_out) noexcept {
    const usize h = x.size() / 2;
    if (a_out.size() < h || d_out.size() < h) {
        return;
    }
    for (usize i = 0; i < h; ++i) {
        a_out[i] = (x[2 * i] + x[2 * i + 1]) * kInvSqrt2;
        d_out[i] = (x[2 * i + 1] - x[2 * i]) * kInvSqrt2;
    }
}

void ihaar_step(ArrayRef<f64> a, ArrayRef<f64> d, MutArrayRef<f64> x_out, bool integer_lifting) noexcept {
    const usize h = (a.size() < d.size()) ? a.size() : d.size();
    if (x_out.size() < 2 * h) {
        return;
    }
    if (integer_lifting) {
        for (usize i = 0; i < h; ++i) {
            const f64 x0 = a[i] - keel::floor(d[i] * 0.5);
            x_out[2 * i] = x0;
            x_out[2 * i + 1] = x0 + d[i];
        }
    } else {
        for (usize i = 0; i < h; ++i) {
            x_out[2 * i] = (a[i] - d[i]) * kInvSqrt2;
            x_out[2 * i + 1] = (a[i] + d[i]) * kInvSqrt2;
        }
    }
}

void ihaar_step_c64(ArrayRef<c64> a, ArrayRef<f64> d, MutArrayRef<c64> x_out) noexcept {
    const usize h = (a.size() < d.size()) ? a.size() : d.size();
    if (x_out.size() < 2 * h) {
        return;
    }
    for (usize i = 0; i < h; ++i) {
        x_out[2 * i] = (a[i] - c64(d[i], 0.0)) * kInvSqrt2;
        x_out[2 * i + 1] = (a[i] + c64(d[i], 0.0)) * kInvSqrt2;
    }
}

void dyaddown_1d(ArrayRef<f64> x, MutArrayRef<f64> out, bool keep_odd) noexcept {
    const usize n = x.size();
    const usize out_n = keep_odd ? ((n + 1) / 2) : (n / 2);
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
    if (n == 0) {
        return;
    }
    const usize out_n = pad_ends ? (2 * n + 1) : (2 * n - 1);
    if (out.size() < out_n) {
        return;
    }
    for (usize k = 0; k < out_n; ++k) {
        out[k] = 0.0;
    }
    const usize start = pad_ends ? 1 : 0;
    for (usize i = 0; i < n; ++i) {
        out[start + 2 * i] = x[i];
    }
}

} // namespace keel::wavelet
