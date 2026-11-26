// keelcpp/wavelet/src/extend.cpp
// Pure zero-STL 1-D signal boundary extension (all 9 wavelet extension modes).

#include "keel/wavelet.h"

#include "keel/arena.h"

namespace keel::wavelet {

namespace {

[[nodiscard]] i64 floor_div(i64 a, i64 b) noexcept {
    i64 q = a / b;
    const i64 r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) {
        --q;
    }
    return q;
}

[[nodiscard]] i64 euclid_mod(i64 a, i64 b) noexcept {
    const i64 r = a % b;
    return (r < 0) ? (r + b) : r;
}

template <typename T>
Array<T> extend_1d_impl(ArrayRef<T> xv, i64 lf, ExtendSide side, ExtendMode mode,
                        Allocator* alloc) {
    Array<T> out(alloc);
    if (xv.empty()) {
        return out;
    }
    if (lf < 0) {
        lf = 0;
    }
    if (xv.size() > as<usize>(Limits<i64>::kMax / 2)) {
        return out;
    }
    const i64 n = as<i64>(xv.size());

    ArenaTemp scratch = get_scratch(alloc);
    Array<T> src(xv, scratch.allocator());
    if (mode == ExtendMode::kPer && (n % 2) == 1) {
        src.push_back(xv[as<usize>(n - 1)]);
    }
    const ArrayRef<T> mid = (mode == ExtendMode::kPer) ? src.as_ref() : xv;
    const i64 mlen = as<i64>(mid.size());

    const i64 left_n = (side == ExtendSide::kRight) ? 0 : lf;
    const i64 right_n = (side == ExtendSide::kLeft) ? 0 : lf;
    const usize ul = as<usize>(left_n);
    const usize um = as<usize>(mlen);
    const usize ur = as<usize>(right_n);
    if (ul > Limits<usize>::kMax - um || ur > Limits<usize>::kMax - (ul + um)) {
        return out;
    }

    auto sample_at = [&](i64 idx) noexcept -> T {
        switch (mode) {
        case ExtendMode::kSym: {
            const i64 period = 2 * n;
            const i64 r = euclid_mod(idx, period);
            return (r < n) ? xv[as<usize>(r)] : xv[as<usize>(period - 1 - r)];
        }
        case ExtendMode::kSymW: {
            if (n == 1) return xv[0];
            const i64 period = 2 * (n - 1);
            const i64 r = euclid_mod(idx, period);
            return (r < n) ? xv[as<usize>(r)] : xv[as<usize>(period - r)];
        }
        case ExtendMode::kAsym: {
            if (n == 1) return xv[0];
            const i64 period = 2 * n;
            const i64 r = euclid_mod(idx, period);
            return (r < n) ? xv[as<usize>(r)] : -xv[as<usize>(period - 1 - r)];
        }
        case ExtendMode::kAsymW: {
            if (n == 1) return xv[0];
            const i64 period = 2 * (n - 1);
            const i64 q = floor_div(idx, period);
            const i64 r = idx - q * period;
            const T base =
                (r < n) ? xv[as<usize>(r)] : (T(2) * xv[as<usize>(n - 1)] - xv[as<usize>(period - r)]);
            const T span = xv[as<usize>(n - 1)] - xv[0];
            return base + T(as<f64>(q) * 2.0) * span;
        }
        case ExtendMode::kSp0:
            return (idx < 0) ? xv[0] : xv[as<usize>(n - 1)];
        case ExtendMode::kSp1:
            if (idx < 0) {
                const T slope = (n >= 2) ? (xv[1] - xv[0]) : T(0);
                return xv[0] + T(as<f64>(idx)) * slope;
            } else {
                const T slope = (n >= 2) ? (xv[as<usize>(n - 1)] - xv[as<usize>(n - 2)]) : T(0);
                return xv[as<usize>(n - 1)] + T(as<f64>(idx - (n - 1))) * slope;
            }
        case ExtendMode::kPer:
            return src[as<usize>(euclid_mod(idx, mlen))];
        case ExtendMode::kPpd:
            return xv[as<usize>(euclid_mod(idx, n))];
        case ExtendMode::kZpd:
            return T(0);
        }
        return T(0);
    };

    out.reserve(ul + um + ur);
    for (i64 i = 0; i < left_n; ++i) {
        out.push_back(sample_at(i - left_n));
    }
    for (i64 i = 0; i < mlen; ++i) {
        out.push_back(mid[as<usize>(i)]);
    }
    for (i64 i = 0; i < right_n; ++i) {
        out.push_back(sample_at(mlen + i));
    }
    return out;
}

} // namespace

Array<f64> wextend_1d(ArrayRef<f64> x, i64 lf, ExtendSide side, ExtendMode mode,
                      Allocator* alloc) {
    return extend_1d_impl<f64>(x, lf, side, mode, alloc);
}

Array<c64> wextend_1d_c64(ArrayRef<c64> x, i64 lf, ExtendSide side, ExtendMode mode,
                          Allocator* alloc) {
    return extend_1d_impl<c64>(x, lf, side, mode, alloc);
}

} // namespace keel::wavelet
