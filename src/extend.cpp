// keelcpp/wavelet/src/extend.cpp
// Pure zero-STL 1-D signal boundary extension (all 9 wavelet extension modes).

#include "keel/wavelet.h"

#include "keel/arena.h"

namespace keel::wavelet {

namespace {

template <typename T>
Array<T> extend_1d_impl(ArrayRef<T> xv, i64 lf, ExtendSide side, ExtendMode mode,
                        Allocator* alloc) {
    const i64 n = as<i64>(xv.size());
    if (lf < 0) {
        lf = 0;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<T> src(xv, scratch.allocator());
    if (mode == ExtendMode::kPer && (n % 2) == 1 && n > 0) {
        src.push_back(xv[as<usize>(n - 1)]);
    }
    const i64 m = as<i64>(src.size());

    auto sample_left = [&](i64 k) noexcept -> T {
        switch (mode) {
        case ExtendMode::kSym:
            return xv[as<usize>(lf - 1 - k)];
        case ExtendMode::kSymW: {
            const i64 idx = lf - k;
            return (idx < n) ? xv[as<usize>(idx)] : xv[as<usize>(n - 1)];
        }
        case ExtendMode::kAsym:
            return -xv[as<usize>(lf - 1 - k)];
        case ExtendMode::kAsymW: {
            const i64 idx = lf - k;
            const T xj = (idx < n) ? xv[as<usize>(idx)] : xv[as<usize>(n - 1)];
            return T(2) * xv[0] - xj;
        }
        case ExtendMode::kSp0:
            return xv[0];
        case ExtendMode::kSp1: {
            const T slope = (n >= 2) ? (xv[1] - xv[0]) : T(0);
            return xv[0] - T(as<f64>(lf - k)) * slope;
        }
        case ExtendMode::kPer:
            return src[as<usize>(m - lf + k)];
        case ExtendMode::kPpd:
            return xv[as<usize>(n - lf + k)];
        case ExtendMode::kZpd:
            return T(0);
        }
        return T(0);
    };

    auto sample_right = [&](i64 k) noexcept -> T {
        switch (mode) {
        case ExtendMode::kSym:
            return xv[as<usize>(n - 1 - k)];
        case ExtendMode::kSymW: {
            const i64 idx = n - 2 - k;
            return (idx >= 0) ? xv[as<usize>(idx)] : xv[0];
        }
        case ExtendMode::kAsym:
            return -xv[as<usize>(n - 1 - k)];
        case ExtendMode::kAsymW: {
            const i64 idx = n - 2 - k;
            const T xj = (idx >= 0) ? xv[as<usize>(idx)] : xv[0];
            return T(2) * xv[as<usize>(n - 1)] - xj;
        }
        case ExtendMode::kSp0:
            return xv[as<usize>(n - 1)];
        case ExtendMode::kSp1: {
            const T slope = (n >= 2) ? (xv[as<usize>(n - 1)] - xv[as<usize>(n - 2)]) : T(0);
            return xv[as<usize>(n - 1)] + T(as<f64>(k + 1)) * slope;
        }
        case ExtendMode::kPer:
            return src[as<usize>(k)];
        case ExtendMode::kPpd:
            return xv[as<usize>(k)];
        case ExtendMode::kZpd:
            return T(0);
        }
        return T(0);
    };

    const i64 left_n = (side == ExtendSide::kRight) ? 0 : lf;
    const i64 right_n = (side == ExtendSide::kLeft) ? 0 : lf;
    const ArrayRef<T> mid = (mode == ExtendMode::kPer) ? src.as_ref() : xv;
    const i64 mlen = as<i64>(mid.size());

    Array<T> out(alloc);
    out.reserve(as<usize>(left_n + mlen + right_n));
    for (i64 i = 0; i < left_n; ++i) {
        out.push_back(sample_left(i));
    }
    for (i64 i = 0; i < mlen; ++i) {
        out.push_back(mid[as<usize>(i)]);
    }
    for (i64 i = 0; i < right_n; ++i) {
        out.push_back(sample_right(i));
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
