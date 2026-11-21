#include "keel/wavelet.h"

namespace keel::wavelet {

namespace {

// In-place heapsort on Array<f64> (zero STL, guaranteed O(n log n)).
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
    for (usize i = 0; i < n; ++i) {
        out[i] = (abs(x[i]) > t) ? x[i] : 0.0;
    }
}

f64 noise_sigma_mad(ArrayRef<f64> detail, Allocator* alloc) {
    const usize n = detail.size();
    if (n == 0) return 0.0;
    Array<f64> tmp(alloc);
    tmp.resize(n);
    for (usize i = 0; i < n; ++i) tmp[i] = abs(detail[i]);
    sort_f64(tmp.data(), n);
    const f64 med = ((n & 1u) == 1u) ? tmp[n / 2] : (0.5 * (tmp[n / 2 - 1] + tmp[n / 2]));
    return med / 0.6745;
}

Array<f64> denoise_visushrink(ArrayRef<f64> x, i32 level, const FilterBank& fb, Allocator* alloc) {
    const usize n = x.size();
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

    Wavedec1Result dec = wavedec(x, use_level, fb, alloc);
    Array<f64> cd1 = detcoef(ArrayRef<f64>(dec.c.data(), dec.c.size()), ArrayRef<usize>(dec.l.data(), dec.l.size()), 1,
                             alloc);
    const f64 sigma = noise_sigma_mad(ArrayRef<f64>(cd1.data(), cd1.size()), alloc);
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
            const f64 s2 = x[i] * x[i];
            if (s2 > 0.0) e -= s2 * log(s2);
        }
        break;
    case EntropyKind::LogEnergy:
        for (usize i = 0; i < n; ++i) {
            const f64 s2 = x[i] * x[i];
            if (s2 > 0.0) e += log(s2);
        }
        break;
    case EntropyKind::Threshold:
        for (usize i = 0; i < n; ++i) {
            if (abs(x[i]) > param) e += 1.0;
        }
        break;
    case EntropyKind::Sure: {
        e = as<f64>(n);
        const f64 t2 = param * param;
        for (usize i = 0; i < n; ++i) {
            if (abs(x[i]) <= param) e -= 2.0;
            e += min(x[i] * x[i], t2);
        }
        break;
    }
    case EntropyKind::Norm:
        for (usize i = 0; i < n; ++i) {
            e += pow(abs(x[i]), param);
        }
        break;
    }
    return e;
}

} // namespace keel::wavelet
