#include <keel/wavelet.h>

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (std::fabs(got - expected) > tol || std::isnan(got)) {
        std::fprintf(stderr, "FAIL: %s (got %.15g, expected %.15g, tol %.3g)\n", msg, got, expected, tol);
        ++g_failures;
    }
}

void test_filters_and_dwt_roundtrip() {
    const char* names[] = {"haar", "db4", "sym4", "coif2", "bior2.4", "rbio4.4"};
    keel::f64 sig[16];
    for (keel::usize i = 0; i < 16; ++i) {
        sig[i] = std::sin(0.3 * static_cast<keel::f64>(i)) + 0.5 * std::cos(0.7 * static_cast<keel::f64>(i));
    }

    for (const char* name : names) {
        keel::wavelet::FilterBank fb;
        check(keel::wavelet::lookup_filters(name, fb), name);

        auto d1 = keel::wavelet::dwt(sig, keel::ArrayRef<keel::f64>(fb.lo_d.data(), fb.lo_d.size()),
                                     keel::ArrayRef<keel::f64>(fb.hi_d.data(), fb.hi_d.size()));
        auto rec = keel::wavelet::idwt(keel::ArrayRef<keel::f64>(d1.ca.data(), d1.ca.size()),
                                       keel::ArrayRef<keel::f64>(d1.cd.data(), d1.cd.size()),
                                       keel::ArrayRef<keel::f64>(fb.lo_r.data(), fb.lo_r.size()),
                                       keel::ArrayRef<keel::f64>(fb.hi_r.data(), fb.hi_r.size()), 16);
        check(rec.size() == 16, "idwt length");
        for (keel::usize i = 0; i < 16; ++i) {
            check_near(rec[i], sig[i], 1e-11, "dwt/idwt roundtrip");
        }

        auto dec = keel::wavelet::wavedec(sig, 3, fb);
        auto wrec = keel::wavelet::waverec(keel::ArrayRef<keel::f64>(dec.c.data(), dec.c.size()),
                                           keel::ArrayRef<keel::usize>(dec.l.data(), dec.l.size()), fb);
        check(wrec.size() == 16, "waverec length");
        for (keel::usize i = 0; i < 16; ++i) {
            check_near(wrec[i], sig[i], 1e-10, "wavedec/waverec roundtrip");
        }
    }
}

void test_dwt2_and_modwt_roundtrip() {
    keel::wavelet::FilterBank fb;
    check(keel::wavelet::lookup_filters("db2", fb), "lookup db2");

    keel::f64 img[6 * 8];
    for (keel::usize i = 0; i < 48; ++i) img[i] = static_cast<keel::f64>(i % 7) - 3.0;
    auto d2 = keel::wavelet::dwt2(img, 6, 8, fb);
    auto r2 = keel::wavelet::idwt2(keel::ArrayRef<keel::f64>(d2.ca.data(), d2.ca.size()),
                                   keel::ArrayRef<keel::f64>(d2.ch.data(), d2.ch.size()),
                                   keel::ArrayRef<keel::f64>(d2.cv.data(), d2.cv.size()),
                                   keel::ArrayRef<keel::f64>(d2.cd.data(), d2.cd.size()), d2.out_rows, d2.out_cols, fb,
                                   6, 8);
    check(r2.rows == 6 && r2.cols == 8, "idwt2 shape");
    for (keel::usize i = 0; i < 48; ++i) {
        check_near(r2.out[i], img[i], 1e-11, "dwt2/idwt2 roundtrip");
    }

    keel::f64 sig[13];
    for (keel::usize i = 0; i < 13; ++i) sig[i] = std::cos(0.4 * static_cast<keel::f64>(i));
    auto w = keel::wavelet::modwt(sig, 3, fb);
    auto irec = keel::wavelet::imodwt(keel::ArrayRef<keel::f64>(w.data(), w.size()), 3, 13, fb);
    check(irec.size() == 13, "imodwt length");
    for (keel::usize i = 0; i < 13; ++i) {
        check_near(irec[i], sig[i], 1e-11, "modwt/imodwt roundtrip");
    }
}

void test_swt_denoise_entropy() {
    keel::wavelet::FilterBank fb;
    check(keel::wavelet::lookup_filters("haar", fb), "lookup haar");

    keel::f64 sig[16];
    for (keel::usize i = 0; i < 16; ++i) sig[i] = static_cast<keel::f64>(i);
    auto s = keel::wavelet::swt_ad(sig, 2, keel::ArrayRef<keel::f64>(fb.lo_d.data(), fb.lo_d.size()),
                                   keel::ArrayRef<keel::f64>(fb.hi_d.data(), fb.hi_d.size()));
    check(s.levels == 2 && s.length == 16, "swt_ad dims");

    auto dn = keel::wavelet::denoise_visushrink(sig, 2, fb);
    check(dn.size() == 16, "denoise length");

    const keel::f64 v[4] = {-2.0, -0.5, 0.5, 3.0};
    keel::f64 out[4];
    keel::wavelet::threshold_soft(v, out, 1.0);
    check_near(out[0], -1.0, 1e-15, "soft -2");
    check_near(out[1], 0.0, 1e-15, "soft -0.5");
    check_near(out[3], 2.0, 1e-15, "soft 3");

    check_near(keel::wavelet::entropy(v, keel::wavelet::EntropyKind::Threshold, 1.0), 2.0, 1e-15, "threshold entropy");
}

} // namespace

int main() {
    test_filters_and_dwt_roundtrip();
    test_dwt2_and_modwt_roundtrip();
    test_swt_denoise_entropy();
    if (g_failures == 0) {
        std::printf("ALL keel::wavelet TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
