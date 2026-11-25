#include <keel/wavelet.h>
#include <keel/format.h>
#include <keel/math.h>

namespace {

keel::i32 g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        keel::eprintln("FAIL: {}", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (keel::abs(got - expected) > tol || keel::is_nan(got)) {
        keel::eprintln("FAIL: {} (got {}, expected {}, tol {})", msg, got, expected, tol);
        ++g_failures;
    }
}

void test_filters_and_dwt_roundtrip() {
    const char* names[] = {"haar", "db4", "sym4", "coif2", "bior2.4", "rbio4.4"};
    keel::f64 sig[16];
    for (keel::usize i = 0; i < 16; ++i) {
        sig[i] = keel::sin(0.3 * keel::as<keel::f64>(i)) + 0.5 * keel::cos(0.7 * keel::as<keel::f64>(i));
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
    for (keel::usize i = 0; i < 48; ++i) img[i] = keel::as<keel::f64>(i % 7) - 3.0;
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
    for (keel::usize i = 0; i < 13; ++i) sig[i] = keel::cos(0.4 * keel::as<keel::f64>(i));
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
    for (keel::usize i = 0; i < 16; ++i) sig[i] = keel::as<keel::f64>(i);
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

void test_multilevel_haar_dyad() {
    keel::wavelet::FilterBank fb;
    check(keel::wavelet::lookup_filters("db3", fb), "lookup db3");

    keel::f64 sig[24];
    for (keel::usize i = 0; i < 24; ++i) {
        sig[i] = keel::sin(0.25 * keel::as<keel::f64>(i)) + 0.3 * keel::as<keel::f64>(i % 5);
    }

    auto dec = keel::wavelet::wavedec(sig, 3, fb);
    keel::f64 ea = 0.0;
    keel::f64 ed[3]{};
    check(keel::wavelet::wenergy(dec.c, dec.l, ea, ed), "wenergy ok");
    check_near(ea + ed[0] + ed[1] + ed[2], 100.0, 1e-10, "wenergy sums to 100%");

    // wrcoef: A3 + D3 + D2 + D1 == original signal
    auto a3 = keel::wavelet::wrcoef(false, dec.c, dec.l, fb.lo_r, fb.hi_r, 3);
    auto d3 = keel::wavelet::wrcoef(true, dec.c, dec.l, fb.lo_r, fb.hi_r, 3);
    auto d2 = keel::wavelet::wrcoef(true, dec.c, dec.l, fb.lo_r, fb.hi_r, 2);
    auto d1 = keel::wavelet::wrcoef(true, dec.c, dec.l, fb.lo_r, fb.hi_r, 1);
    check(a3.size() == 24 && d1.size() == 24, "wrcoef length");
    for (keel::usize i = 0; i < 24; ++i) {
        check_near(a3[i] + d3[i] + d2[i] + d1[i], sig[i], 1e-10, "wrcoef perfect sum");
    }

    // upcoef
    const keel::f64 impulse[1] = {1.0};
    auto up = keel::wavelet::upcoef(false, impulse, fb, 2);
    check(up.size() > 0, "upcoef non-empty");
    check(keel::wavelet::wmaxlev(64, 6) == 3, "wmaxlev(64, db3)");

    // 2D multilevel roundtrip
    keel::f64 img[10 * 12];
    for (keel::usize i = 0; i < 120; ++i) {
        img[i] = keel::sin(0.1 * keel::as<keel::f64>(i)) + keel::as<keel::f64>(i % 9);
    }
    auto dec2 = keel::wavelet::wavedec2(img, 10, 12, 2, fb);
    auto rec2 = keel::wavelet::waverec2(dec2.c, dec2.s_rows, dec2.s_cols, fb);
    check(rec2.rows == 10 && rec2.cols == 12, "waverec2 shape");
    for (keel::usize i = 0; i < 120; ++i) {
        check_near(rec2.out[i], img[i], 1e-9, "wavedec2/waverec2 roundtrip");
    }
    auto dh1 = keel::wavelet::detcoef2(keel::wavelet::Detail2D::Horizontal, dec2.c, dec2.s_rows, dec2.s_cols, 1);
    check(dh1.rows == dec2.s_rows[2] && dh1.cols == dec2.s_cols[2], "detcoef2 shape");

    // Haar orthogonal & integer lifting roundtrip
    const keel::f64 hx[8] = {3.0, 7.0, -2.0, 5.0, 11.0, 0.0, 4.0, -8.0};
    keel::f64 ha[4]{}, hd[4]{}, hrec[8]{};
    check(keel::wavelet::haar_max_level(8) == 3, "haar_max_level(8)");
    keel::wavelet::haar_step(hx, ha, hd, false);
    keel::wavelet::ihaar_step(ha, hd, hrec, false);
    for (keel::usize i = 0; i < 8; ++i) check_near(hrec[i], hx[i], 1e-14, "haar orth roundtrip");

    keel::wavelet::haar_step(hx, ha, hd, true);
    keel::wavelet::ihaar_step(ha, hd, hrec, true);
    for (keel::usize i = 0; i < 8; ++i) check_near(hrec[i], hx[i], 1e-15, "haar integer lifting roundtrip");

    // Dyadic down/up
    keel::f64 dd[4]{}, du[7]{};
    keel::wavelet::dyaddown_1d(hx, dd, false);
    check_near(dd[0], 7.0, 1e-15, "dyaddown even 0");
    keel::wavelet::dyadup_1d(dd, du, false);
    check_near(du[0], 7.0, 1e-15, "dyadup 0");
    check_near(du[1], 0.0, 1e-15, "dyadup 1");
    check_near(du[2], 5.0, 1e-15, "dyadup 2");
}

void test_shapes_centfrq_cwt() {
    keel::f64 x[11]{}, psi[11]{};
    keel::wavelet::mexihat(-5.0, 5.0, psi, x);
    check_near(x[5], 0.0, 1e-15, "mexihat center x");
    check(psi[5] > 0.8, "mexihat peak > 0.8");

    keel::wavelet::morlet(-4.0, 4.0, psi, x);
    check_near(psi[5], 1.0, 1e-15, "morlet(0) == 1");
    check_near(keel::wavelet::meyeraux(0.5), 0.5, 1e-15, "meyeraux(0.5) == 0.5");

    keel::wavelet::gauswavf(-5.0, 5.0, 2, psi, x);
    check(keel::is_finite(psi[5]), "gauswavf finite");

    keel::c64 cpsi[11]{};
    keel::wavelet::cgauwavf(-5.0, 5.0, 1, cpsi, x);
    check(keel::abs(cpsi[5]) > 0.0, "cgauwavf non-zero");

    auto cf_morl = keel::wavelet::centfrq("morl", 8, true);
    check(cf_morl.ok, "centfrq morl ok");
    check_near(cf_morl.freq, 0.8125, 0.05, "centfrq morl ~0.8125");
    check(cf_morl.recfreq_real.size() == cf_morl.xval.size(), "centfrq recfreq size");

    auto cf_db4 = keel::wavelet::centfrq("db4", 8, false);
    check(cf_db4.ok && cf_db4.freq > 0.5 && cf_db4.freq < 1.0, "centfrq db4");

    // CWT peak frequency check on a 50 Hz tone sampled at 500 Hz
    constexpr keel::f64 kFs = 500.0;
    constexpr keel::usize kN = 256;
    keel::Array<keel::f64> tone(kN);
    for (keel::usize i = 0; i < kN; ++i) {
        tone[i] = keel::sin(keel::kTau * 50.0 * keel::as<keel::f64>(i) / kFs);
    }
    const keel::wavelet::CwtWavelet waves[3] = {
        keel::wavelet::CwtWavelet::Morse,
        keel::wavelet::CwtWavelet::Amor,
        keel::wavelet::CwtWavelet::Bump,
    };
    for (keel::usize w = 0; w < 3; ++w) {
        auto res = keel::wavelet::cwt(tone, waves[w], kFs, 10);
        check(res.num_scales > 5 && res.num_samples == kN, "cwt dims");
        // Check peak scale at middle sample kN/2
        keel::usize best_s = 0;
        keel::f64 best_mag = -1.0;
        for (keel::usize s = 0; s < res.num_scales; ++s) {
            const keel::f64 m = keel::abs(res.cfs_colmajor[s + (kN / 2) * res.num_scales]);
            if (m > best_mag) {
                best_mag = m;
                best_s = s;
            }
        }
        check_near(res.freqs[best_s], 50.0, 5.0, "cwt peak frequency ~50 Hz");
    }
}

void test_wextend() {
    const keel::f64 x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto sym = keel::wavelet::wextend_1d(x, 2, keel::wavelet::ExtendSide::kBoth,
                                         keel::wavelet::ExtendMode::kSym);
    check(sym.size() == 9 && sym[0] == 2.0 && sym[1] == 1.0 && sym[7] == 5.0 && sym[8] == 4.0,
          "wextend sym");

    auto symw = keel::wavelet::wextend_1d(x, 2, keel::wavelet::ExtendSide::kBoth,
                                          keel::wavelet::ExtendMode::kSymW);
    check(symw.size() == 9 && symw[0] == 3.0 && symw[1] == 2.0 && symw[7] == 4.0 && symw[8] == 3.0,
          "wextend symw");

    auto sp1 = keel::wavelet::wextend_1d(x, 2, keel::wavelet::ExtendSide::kBoth,
                                         keel::wavelet::ExtendMode::kSp1);
    check(sp1.size() == 9 && sp1[0] == -1.0 && sp1[1] == 0.0 && sp1[7] == 6.0 && sp1[8] == 7.0,
          "wextend sp1");

    auto per = keel::wavelet::wextend_1d(x, 2, keel::wavelet::ExtendSide::kBoth,
                                         keel::wavelet::ExtendMode::kPer);
    check(per.size() == 10 && per[0] == 5.0 && per[1] == 5.0 && per[7] == 5.0 && per[8] == 1.0,
          "wextend per odd pad");

    const keel::c64 xc[3] = {keel::c64(1.0, 2.0), keel::c64(3.0, 4.0), keel::c64(5.0, 6.0)};
    auto zc = keel::wavelet::wextend_1d_c64(xc, 1, keel::wavelet::ExtendSide::kLeft,
                                            keel::wavelet::ExtendMode::kZpd);
    check(zc.size() == 4 && zc[0] == keel::c64(0.0, 0.0) && zc[1] == keel::c64(1.0, 2.0),
          "wextend c64 zpd left");
}

} // namespace

int main() {
    test_filters_and_dwt_roundtrip();
    test_dwt2_and_modwt_roundtrip();
    test_swt_denoise_entropy();
    test_multilevel_haar_dyad();
    test_shapes_centfrq_cwt();
    test_wextend();
    if (g_failures == 0) {
        keel::println("ALL keel::wavelet TESTS PASSED");
        return 0;
    }
    keel::eprintln("{} FAILURE(S)", g_failures);
    return 1;
}
