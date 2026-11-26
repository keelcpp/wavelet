#include "keel/wavelet.h"

#include "keel/arena.h"

namespace keel::wavelet {

namespace {

// In-place heapsort on finite f64 values (zero STL, guaranteed O(n log n)).
void sort_f64(f64* data, usize n) noexcept {
    if (n < 2) return;
    auto sift_down = [data](usize start, usize end) {
        usize root = start;
        while (2 * root + 1 < end) {
            usize child = 2 * root + 1;
            if (child + 1 < end && data[child] < data[child + 1]) ++child;
            if (data[root] < data[child]) {
                swap(data[root], data[child]);
                root = child;
            } else {
                break;
            }
        }
    };
    for (usize i = n / 2; i > 0; --i) sift_down(i - 1, n);
    for (usize end = n - 1; end > 0; --end) {
        swap(data[0], data[end]);
        sift_down(0, end);
    }
}

[[nodiscard]] f64 soft_scalar(f64 x, f64 t) noexcept {
    if (is_nan(x) || is_nan(t)) return kNaN;
    if (x == 0.0) return 0.0;
    const f64 ax = abs(x);
    const f64 mag = ax - t;
    if (mag <= 0.0) return 0.0;
    return signbit(x) ? -mag : mag;
}

} // namespace

void threshold_soft(ArrayRef<f64> x, MutArrayRef<f64> out, f64 t) noexcept {
    const usize n = min(x.size(), out.size());
    for (usize i = 0; i < n; ++i) {
        out[i] = soft_scalar(x[i], t);
    }
}

void threshold_hard(ArrayRef<f64> x, MutArrayRef<f64> out, f64 t) noexcept {
    const usize n = min(x.size(), out.size());
    if (is_nan(t)) {
        for (usize i = 0; i < n; ++i) out[i] = kNaN;
        return;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 xi = x[i];
        if (is_nan(xi)) {
            out[i] = kNaN;
        } else {
            out[i] = (abs(xi) > t) ? xi : 0.0;
        }
    }
}

f64 noise_sigma_mad(ArrayRef<f64> detail, Allocator* alloc) {
    const usize n = detail.size();
    if (n == 0) return 0.0;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> tmp(scratch.allocator());
    tmp.resize(n);
    for (usize i = 0; i < n; ++i) {
        if (is_nan(detail[i])) return kNaN;
        tmp[i] = abs(detail[i]);
    }
    sort_f64(tmp.data(), n);
    const f64 med = ((n & 1u) == 1u) ? tmp[n / 2] : (0.5 * tmp[n / 2 - 1] + 0.5 * tmp[n / 2]);
    return med / 0.6745;
}

Array<f64> denoise_visushrink(ArrayRef<f64> x, i32 level, const FilterBank& fb, Allocator* alloc) {
    const usize n = x.size();
    if (fb.lo_d.size() < 2 || fb.hi_d.size() != fb.lo_d.size() || fb.lo_r.size() < 2 ||
        fb.hi_r.size() != fb.lo_r.size()) {
        return Array<f64>(alloc);
    }
    if (n < 2) {
        Array<f64> copy(alloc);
        copy.resize(n);
        for (usize i = 0; i < n; ++i) copy[i] = x[i];
        return copy;
    }

    i32 use_level = level;
    if (use_level <= 0) {
        const i32 max_log = as<i32>(floor(log2(as<f64>(n))));
        use_level = min(max_log, as<i32>(5));
        if (use_level < 1) use_level = 1;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Wavedec1Result dec = wavedec(x, use_level, fb, scratch.allocator());
    if (dec.l.size() < as<usize>(use_level) + 2) {
        return Array<f64>(alloc);
    }
    Array<f64> cd1 = detcoef(ArrayRef<f64>(dec.c.data(), dec.c.size()), ArrayRef<usize>(dec.l.data(), dec.l.size()), 1,
                             scratch.allocator());
    const f64 sigma = noise_sigma_mad(ArrayRef<f64>(cd1.data(), cd1.size()), scratch.allocator());
    const f64 thr = sigma * sqrt(2.0 * log(as<f64>(n)));

    usize off = dec.l[0];
    for (i32 k = 0; k < use_level; ++k) {
        const usize d_len = dec.l[1 + as<usize>(k)];
        for (usize i = 0; i < d_len && off + i < dec.c.size(); ++i) {
            dec.c[off + i] = soft_scalar(dec.c[off + i], thr);
        }
        off += d_len;
    }

    return waverec(ArrayRef<f64>(dec.c.data(), dec.c.size()), ArrayRef<usize>(dec.l.data(), dec.l.size()), fb, alloc);
}

f64 entropy(ArrayRef<f64> x, EntropyKind kind, f64 param) noexcept {
    const usize n = x.size();
    f64 e = 0.0;
    switch (kind) {
    case EntropyKind::Shannon:
        for (usize i = 0; i < n; ++i) {
            const f64 xi = x[i];
            if (is_nan(xi)) return kNaN;
            const f64 s2 = xi * xi;
            if (s2 > 0.0) e -= s2 * log(s2);
        }
        break;
    case EntropyKind::LogEnergy:
        for (usize i = 0; i < n; ++i) {
            const f64 xi = x[i];
            if (is_nan(xi)) return kNaN;
            const f64 ax = abs(xi);
            if (ax > 0.0) e += 2.0 * log(ax);
        }
        break;
    case EntropyKind::Threshold:
        if (is_nan(param)) return kNaN;
        for (usize i = 0; i < n; ++i) {
            const f64 xi = x[i];
            if (is_nan(xi)) return kNaN;
            if (abs(xi) > param) e += 1.0;
        }
        break;
    case EntropyKind::Sure: {
        if (is_nan(param)) return kNaN;
        e = as<f64>(n);
        const f64 t2 = param * param;
        for (usize i = 0; i < n; ++i) {
            const f64 xi = x[i];
            if (is_nan(xi)) return kNaN;
            if (abs(xi) <= param) e -= 2.0;
            const f64 s2 = xi * xi;
            e += (s2 < t2) ? s2 : t2;
        }
        break;
    }
    case EntropyKind::Norm:
        if (is_nan(param)) return kNaN;
        for (usize i = 0; i < n; ++i) {
            const f64 xi = x[i];
            if (is_nan(xi)) return kNaN;
            e += pow(abs(xi), param);
        }
        break;
    }
    return e;
}

} // namespace keel::wavelet
