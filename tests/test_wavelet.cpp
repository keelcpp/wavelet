#include <keel/format.h>
#include <keel/fs.h>
#include <keel/math.h>
#include <keel/wavelet.h>

namespace {

keel::i32 g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        keel::eprintln("FAIL: %s", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (keel::abs(got - expected) > tol || keel::is_nan(got)) {
        keel::eprintln("FAIL: %s (got %.15g, expected %.15g, tol %.15g)", msg, got, expected, tol);
        ++g_failures;
    }
}

constexpr const char* kAll55Wavelets[] = {
    "haar",    "db1",     "db2",     "db3",     "db4",     "db5",     "db6",     "db7",     "db8",     "db9",
    "db10",    "sym2",    "sym3",    "sym4",    "sym5",    "sym6",    "sym7",    "sym8",    "sym9",    "sym10",
    "coif1",   "coif2",   "coif3",   "coif4",   "coif5",   "bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4",
    "bior2.6", "bior2.8", "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9", "bior4.4", "bior5.5", "bior6.8",
    "rbio1.1", "rbio1.3", "rbio1.5", "rbio2.2", "rbio2.4", "rbio2.6", "rbio2.8", "rbio3.1", "rbio3.3", "rbio3.5",
    "rbio3.7", "rbio3.9", "rbio4.4", "rbio5.5", "rbio6.8",
};

void test_filters_and_dwt_roundtrip() {
    keel::f64 sig16[16];
    for (keel::usize i = 0; i < 16; ++i) {
        sig16[i] = keel::sin(0.3 * keel::as<keel::f64>(i)) + 0.5 * keel::cos(0.7 * keel::as<keel::f64>(i));
    }
    const keel::f64 sig15[15] = {1.2, -0.5, 3.1, 0.0, -2.4, 1.7, 0.9, -1.1, 2.2, 0.3, -0.8, 1.5, -2.0, 0.4, 1.9};
    const keel::f64 sig3[3] = {1.0, 2.0, 3.0};
    const keel::f64 sig1[1] = {4.25};

    // All 55 built-in filter banks: roundtrip on even (16), odd (15), short (3 < lf), and length-1 signals.
    for (const char* name : kAll55Wavelets) {
        const bool is_coif5 = (keel::StringRef(name) == "coif5");
        const keel::f64 tol1 = is_coif5 ? 1e-8 : 5e-10;
        const keel::f64 tol3 = is_coif5 ? 5e-8 : 1e-9;

        keel::wavelet::FilterBank fb;
        check(keel::wavelet::lookup_filters(name, fb), name);
        check(!fb.lo_d.empty() && fb.lo_d.size() == fb.hi_d.size() && fb.lo_d.size() == fb.lo_r.size() &&
                  fb.lo_d.size() == fb.hi_r.size(),
              "filter bank sizes match");

        // 16-sample 1-level and 3-level roundtrip
        auto d1 = keel::wavelet::dwt(sig16, fb.lo_d, fb.hi_d);
        auto rec = keel::wavelet::idwt(d1.ca, d1.cd, fb.lo_r, fb.hi_r, 16);
        check(rec.size() == 16, "idwt16 length");
        for (keel::usize i = 0; i < 16; ++i) {
            check_near(rec[i], sig16[i], tol1, "dwt/idwt16 roundtrip");
        }

        auto dec = keel::wavelet::wavedec(sig16, 3, fb);
        auto wrec = keel::wavelet::waverec(dec.c, dec.l, fb);
        check(wrec.size() == 16, "waverec16 length");
        for (keel::usize i = 0; i < 16; ++i) {
            check_near(wrec[i], sig16[i], tol3, "wavedec/waverec16 roundtrip");
        }

        // Odd 15-sample roundtrip
        auto d15 = keel::wavelet::dwt(sig15, fb.lo_d, fb.hi_d);
        auto r15 = keel::wavelet::idwt(d15.ca, d15.cd, fb.lo_r, fb.hi_r, 15);
        check(r15.size() == 15, "idwt15 length");
        for (keel::usize i = 0; i < 15; ++i) {
            check_near(r15[i], sig15[i], tol1, "dwt/idwt15 odd roundtrip");
        }

        // Short signal (n=3 < lf for most wavelets) roundtrip (regression for sym_extend periodic reflection)
        auto d3 = keel::wavelet::dwt(sig3, fb.lo_d, fb.hi_d);
        auto r3 = keel::wavelet::idwt(d3.ca, d3.cd, fb.lo_r, fb.hi_r, 3);
        check(r3.size() == 3, "idwt3 length");
        for (keel::usize i = 0; i < 3; ++i) {
            check_near(r3[i], sig3[i], tol1, "dwt/idwt3 short roundtrip");
        }

        // Length-1 signal roundtrip
        auto d_one = keel::wavelet::dwt(sig1, fb.lo_d, fb.hi_d);
        auto r_one = keel::wavelet::idwt(d_one.ca, d_one.cd, fb.lo_r, fb.hi_r, 1);
        check(r_one.size() == 1, "idwt1 length");
        check_near(r_one[0], sig1[0], tol1, "dwt/idwt1 roundtrip");
    }

    // Unknown filter lookup
    keel::wavelet::FilterBank bad_fb;
    check(!keel::wavelet::lookup_filters("nonexistent", bad_fb), "unknown filter false");
    check(!keel::wavelet::lookup_filters("", bad_fb), "empty filter false");

    // orthfilt + qmf (including in-place aliasing & negative/extreme parity)
    const keel::f64 w_haar[2] = {0.5, 0.5};
    auto ofb = keel::wavelet::orthfilt(w_haar);
    check(ofb.lo_r.size() == 2, "orthfilt size");
    check_near(ofb.lo_r[0], keel::kInvSqrt2, 1e-15, "orthfilt lo_r[0]");
    check_near(ofb.hi_d[0], -keel::kInvSqrt2, 1e-15, "orthfilt hi_d[0]");
    check_near(ofb.hi_d[1], keel::kInvSqrt2, 1e-15, "orthfilt hi_d[1]");

    keel::f64 q_even[4] = {1.0, 2.0, 3.0, 4.0};
    keel::wavelet::qmf(q_even, q_even, 0); // in-place aliasing!
    check_near(q_even[0], 4.0, 1e-15, "qmf inplace even p=0 [0]");
    check_near(q_even[1], -3.0, 1e-15, "qmf inplace even p=0 [1]");
    check_near(q_even[2], 2.0, 1e-15, "qmf inplace even p=0 [2]");
    check_near(q_even[3], -1.0, 1e-15, "qmf inplace even p=0 [3]");

    keel::f64 q_odd[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    keel::wavelet::qmf(q_odd, q_odd, -1); // in-place odd length, negative odd parity
    check_near(q_odd[0], -5.0, 1e-15, "qmf inplace odd p=-1 [0]");
    check_near(q_odd[1], 4.0, 1e-15, "qmf inplace odd p=-1 [1]");
    check_near(q_odd[2], -3.0, 1e-15, "qmf inplace odd p=-1 [2]");
    check_near(q_odd[3], 2.0, 1e-15, "qmf inplace odd p=-1 [3]");
    check_near(q_odd[4], -1.0, 1e-15, "qmf inplace odd p=-1 [4]");

    keel::f64 q_min[4] = {1.0, 2.0, 3.0, 4.0};
    keel::wavelet::qmf(q_min, q_min, keel::Limits<keel::i32>::kMin);
    check_near(q_min[0], 4.0, 1e-15, "qmf kMin parity [0]");
    check_near(q_min[1], -3.0, 1e-15, "qmf kMin parity [1]");

    // idwt single-branch (empty ca or empty cd) regression: default target length must use max(la, ld)
    keel::wavelet::FilterBank db4;
    check(keel::wavelet::lookup_filters("db4", db4), "lookup db4");
    auto d_s = keel::wavelet::dwt(sig16, db4.lo_d, db4.hi_d);
    auto rec_a = keel::wavelet::idwt(d_s.ca, keel::ArrayRef<keel::f64>{}, db4.lo_r, db4.hi_r);
    auto rec_d = keel::wavelet::idwt(keel::ArrayRef<keel::f64>{}, d_s.cd, db4.lo_r, db4.hi_r);
    check(rec_a.size() == 16 && rec_d.size() == 16, "idwt single-branch default length == 16");
    for (keel::usize i = 0; i < 16; ++i) {
        check_near(rec_a[i] + rec_d[i], sig16[i], 1e-11, "idwt(ca,[]) + idwt([],cd) == x");
    }
    auto rec_zero = keel::wavelet::idwt(d_s.ca, d_s.cd, db4.lo_r, db4.hi_r, 0);
    check(rec_zero.empty(), "idwt target_len=0 returns empty");

    // wavedec/appcoef/detcoef invalid inputs & overflow protection
    keel::wavelet::FilterBank empty_fb;
    auto bad_dec = keel::wavelet::wavedec(sig16, 2, empty_fb);
    check(bad_dec.c.empty() && bad_dec.l.empty(), "wavedec invalid fb returns empty");
    auto bad_dec2 = keel::wavelet::wavedec(keel::ArrayRef<keel::f64>{}, 2, db4);
    check(bad_dec2.c.empty() && bad_dec2.l.empty(), "wavedec empty x returns empty");

    const keel::usize bad_l[3] = {4, keel::Limits<keel::usize>::kMax, 8};
    auto bad_det = keel::wavelet::detcoef(sig16, bad_l, 1);
    check(bad_det.empty(), "detcoef overflow l rejected");
    auto bad_app = keel::wavelet::appcoef(sig16, bad_l, db4.lo_r, db4.hi_r, 0);
    check(bad_app.empty(), "appcoef overflow l rejected");
    auto dec_ok = keel::wavelet::wavedec(sig16, 2, db4);
    auto bad_app_filt = keel::wavelet::appcoef(dec_ok.c, dec_ok.l, keel::ArrayRef<keel::f64>{}, db4.hi_r, 0);
    check(bad_app_filt.empty(), "appcoef invalid filter rejected");
}

void test_dwt2_and_modwt_roundtrip() {
    keel::wavelet::FilterBank fb;
    check(keel::wavelet::lookup_filters("db2", fb), "lookup db2");

    keel::f64 img[6 * 8];
    for (keel::usize i = 0; i < 48; ++i) img[i] = keel::as<keel::f64>(i % 7) - 3.0;
    auto d2 = keel::wavelet::dwt2(img, 6, 8, fb);
    auto r2 = keel::wavelet::idwt2(d2.ca, d2.ch, d2.cv, d2.cd, d2.out_rows, d2.out_cols, fb, 6, 8);
    check(r2.rows == 6 && r2.cols == 8, "idwt2 shape");
    for (keel::usize i = 0; i < 48; ++i) {
        check_near(r2.out[i], img[i], 1e-11, "dwt2/idwt2 roundtrip");
    }

    // Odd and short 2D dimensions (5x7 and 3x2 with db4)
    keel::wavelet::FilterBank db4;
    check(keel::wavelet::lookup_filters("db4", db4), "lookup db4");
    keel::f64 img_odd[5 * 7];
    for (keel::usize i = 0; i < 35; ++i) img_odd[i] = keel::sin(0.5 * keel::as<keel::f64>(i));
    auto d2_odd = keel::wavelet::dwt2(img_odd, 5, 7, db4);
    auto r2_odd =
        keel::wavelet::idwt2(d2_odd.ca, d2_odd.ch, d2_odd.cv, d2_odd.cd, d2_odd.out_rows, d2_odd.out_cols, db4, 5, 7);
    check(r2_odd.rows == 5 && r2_odd.cols == 7, "idwt2 odd shape");
    for (keel::usize i = 0; i < 35; ++i) {
        check_near(r2_odd.out[i], img_odd[i], 1e-10, "dwt2/idwt2 odd 5x7 db4 roundtrip");
    }

    auto r2_zero = keel::wavelet::idwt2(d2.ca, d2.ch, d2.cv, d2.cd, d2.out_rows, d2.out_cols, fb, 0, 8);
    check(r2_zero.rows == 0 && r2_zero.cols == 0 && r2_zero.out.empty(), "idwt2 target_rows=0");

    // Overflow guard on rows * cols
    auto d2_ovf = keel::wavelet::dwt2(img, keel::Limits<keel::usize>::kMax, 2, fb);
    check(d2_ovf.out_rows == 0 && d2_ovf.out_cols == 0, "dwt2 overflow guard");

    // MODWT roundtrip & energy conservation & high level (>= 64, no shift UB!)
    keel::f64 sig[13];
    for (keel::usize i = 0; i < 13; ++i) sig[i] = keel::cos(0.4 * keel::as<keel::f64>(i));
    auto w = keel::wavelet::modwt(sig, 3, fb);
    auto irec = keel::wavelet::imodwt(w, 3, 13, fb);
    check(irec.size() == 13, "imodwt length");
    for (keel::usize i = 0; i < 13; ++i) {
        check_near(irec[i], sig[i], 1e-11, "modwt/imodwt roundtrip");
    }

    // Energy conservation: ||W||_2^2 == ||x||_2^2
    keel::f64 ex = 0.0, ew = 0.0;
    for (keel::usize i = 0; i < 13; ++i) ex += sig[i] * sig[i];
    for (keel::usize i = 0; i < w.size(); ++i) ew += w[i] * w[i];
    check_near(ew, ex, 1e-10, "modwt energy conservation");

    // High level >= 64 regression (no shift UB)
    auto w70 = keel::wavelet::modwt(sig, 70, fb);
    check(w70.size() == 71 * 13, "modwt level 70 size");
    auto irec70 = keel::wavelet::imodwt(w70, 70, 13, fb);
    check(irec70.size() == 13, "imodwt level 70 size");
    for (keel::usize i = 0; i < 13; ++i) {
        check_near(irec70[i], sig[i], 1e-9, "modwt/imodwt level 70 roundtrip");
    }

    // Mismatched filter sizes rejected in modwt/imodwt
    keel::wavelet::FilterBank mismatch_fb;
    mismatch_fb.lo_d.resize(4);
    mismatch_fb.hi_d.resize(2);
    check(keel::wavelet::modwt(sig, 2, mismatch_fb).empty(), "modwt mismatched filters rejected");
    check(keel::wavelet::imodwt(w, 3, 13, mismatch_fb).empty(), "imodwt mismatched filters rejected");
    check(keel::wavelet::imodwt(w, keel::Limits<keel::usize>::kMax, 13, fb).empty(), "imodwt levels=kMax rejected");
}

void test_swt_denoise_entropy() {
    keel::wavelet::FilterBank fb;
    check(keel::wavelet::lookup_filters("haar", fb), "lookup haar");

    keel::f64 sig[16];
    for (keel::usize i = 0; i < 16; ++i) sig[i] = keel::as<keel::f64>(i);
    auto s = keel::wavelet::swt_ad(sig, 2, fb.lo_d, fb.hi_d);
    check(s.levels == 2 && s.length == 16 && s.swa_colmajor.size() == 32 && s.swd_colmajor.size() == 32,
          "swt_ad dims");

    // Non-divisible length & level >= 64 regression: must return levels==0, length==0 and no shift UB
    auto s_bad = keel::wavelet::swt_ad(keel::ArrayRef<keel::f64>(sig, 15), 2, fb.lo_d, fb.hi_d);
    check(s_bad.levels == 0 && s_bad.length == 0 && s_bad.swa_colmajor.empty(), "swt_ad non-divisible empty");
    auto s_hi = keel::wavelet::swt_ad(sig, 64, fb.lo_d, fb.hi_d);
    check(s_hi.levels == 0 && s_hi.length == 0 && s_hi.swa_colmajor.empty(), "swt_ad level=64 empty");

    auto dn = keel::wavelet::denoise_visushrink(sig, 2, fb);
    check(dn.size() == 16, "denoise length");
    keel::wavelet::FilterBank bad_fb;
    check(keel::wavelet::denoise_visushrink(sig, 2, bad_fb).empty(), "denoise invalid fb empty");

    const keel::f64 v[4] = {-2.0, -0.5, 0.5, 3.0};
    keel::f64 out[4];
    keel::wavelet::threshold_soft(v, out, 1.0);
    check_near(out[0], -1.0, 1e-15, "soft -2");
    check_near(out[1], 0.0, 1e-15, "soft -0.5");
    check_near(out[3], 2.0, 1e-15, "soft 3");

    // NaN propagation in threshold_soft, threshold_hard, noise_sigma_mad, and entropy
    const keel::f64 v_nan[3] = {2.0, keel::kNaN, -3.0};
    keel::f64 out_nan[3];
    keel::wavelet::threshold_soft(v_nan, out_nan, 1.0);
    check(keel::is_nan(out_nan[1]), "threshold_soft propagates NaN");
    keel::wavelet::threshold_hard(v_nan, out_nan, 1.0);
    check(keel::is_nan(out_nan[1]), "threshold_hard propagates NaN");
    keel::wavelet::threshold_hard(v, out, keel::kNaN);
    check(keel::is_nan(out[0]), "threshold_hard NaN threshold");

    const keel::f64 zero_in[1] = {0.0};
    keel::f64 zero_out[1];
    keel::wavelet::threshold_soft(zero_in, zero_out, -1.0);
    check_near(zero_out[0], 0.0, 1e-15, "threshold_soft(0, -1) == 0");

    check(keel::is_nan(keel::wavelet::noise_sigma_mad(v_nan)), "noise_sigma_mad propagates NaN");
    const keel::f64 huge_pair[2] = {1e308, 1e308};
    check(keel::is_finite(keel::wavelet::noise_sigma_mad(huge_pair)), "noise_sigma_mad no midpoint overflow");

    check_near(keel::wavelet::entropy(v, keel::wavelet::EntropyKind::Threshold, 1.0), 2.0, 1e-15, "threshold entropy");
    check(keel::is_nan(keel::wavelet::entropy(v_nan, keel::wavelet::EntropyKind::Shannon)), "entropy Shannon NaN");
    check(keel::is_nan(keel::wavelet::entropy(v_nan, keel::wavelet::EntropyKind::LogEnergy)), "entropy LogEnergy NaN");
    check(keel::is_nan(keel::wavelet::entropy(v_nan, keel::wavelet::EntropyKind::Threshold, 1.0)),
          "entropy Threshold NaN");
    check(keel::is_nan(keel::wavelet::entropy(v_nan, keel::wavelet::EntropyKind::Sure, 1.0)), "entropy Sure NaN");
    check(keel::is_nan(keel::wavelet::entropy(v_nan, keel::wavelet::EntropyKind::Norm, 2.0)), "entropy Norm NaN");
    check(keel::is_nan(keel::wavelet::entropy(v, keel::wavelet::EntropyKind::Sure, keel::kNaN)),
          "entropy Sure NaN param");

    const keel::f64 huge_e[1] = {1e200};
    check(keel::is_finite(keel::wavelet::entropy(huge_e, keel::wavelet::EntropyKind::LogEnergy)),
          "entropy LogEnergy 1e200 finite");
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

    // wenergy overflow/underflow safety (1e200 and 1e-200)
    keel::Array<keel::f64> c_big(dec.c.size());
    keel::Array<keel::f64> c_tiny(dec.c.size());
    for (keel::usize i = 0; i < dec.c.size(); ++i) {
        c_big[i] = dec.c[i] * 1e200;
        c_tiny[i] = dec.c[i] * 1e-200;
    }
    keel::f64 ea_b = 0.0, ea_t = 0.0;
    keel::f64 ed_b[3]{}, ed_t[3]{};
    check(keel::wavelet::wenergy(c_big, dec.l, ea_b, ed_b), "wenergy 1e200 ok");
    check_near(ea_b, ea, 1e-10, "wenergy 1e200 ea matches");
    check_near(ea_b + ed_b[0] + ed_b[1] + ed_b[2], 100.0, 1e-10, "wenergy 1e200 sums to 100%");
    check(keel::wavelet::wenergy(c_tiny, dec.l, ea_t, ed_t), "wenergy 1e-200 ok");
    check_near(ea_t, ea, 1e-10, "wenergy 1e-200 ea matches");
    check_near(ea_t + ed_t[0] + ed_t[1] + ed_t[2], 100.0, 1e-10, "wenergy 1e-200 sums to 100%");

    // wrcoef: A3 + D3 + D2 + D1 == original signal
    auto a3 = keel::wavelet::wrcoef(false, dec.c, dec.l, fb.lo_r, fb.hi_r, 3);
    auto d3 = keel::wavelet::wrcoef(true, dec.c, dec.l, fb.lo_r, fb.hi_r, 3);
    auto d2 = keel::wavelet::wrcoef(true, dec.c, dec.l, fb.lo_r, fb.hi_r, 2);
    auto d1 = keel::wavelet::wrcoef(true, dec.c, dec.l, fb.lo_r, fb.hi_r, 1);
    check(a3.size() == 24 && d1.size() == 24, "wrcoef length");
    for (keel::usize i = 0; i < 24; ++i) {
        check_near(a3[i] + d3[i] + d2[i] + d1[i], sig[i], 1e-10, "wrcoef perfect sum");
    }

    // upcoef & exact integer wmaxlev
    const keel::f64 impulse[1] = {1.0};
    auto up = keel::wavelet::upcoef(false, impulse, fb, 2);
    check(up.size() > 0, "upcoef non-empty");
    auto up0 = keel::wavelet::upcoef(false, sig, fb, 0, 10);
    check(up0.size() == 10, "upcoef level 0 crop");
    check(keel::wavelet::wmaxlev(64, 6) == 3, "wmaxlev(64, db3)");
    check(keel::wavelet::wmaxlev(10, 6) == 1, "wmaxlev(10, db3) exact boundary");
    check(keel::wavelet::wmaxlev(9, 6) == 0, "wmaxlev(9, db3)");
    check(keel::wavelet::wmaxlev(keel::as<keel::usize>(1) << 55, 2) == 55, "wmaxlev(2^55, haar)");

    // 2D multilevel roundtrip (including odd 9x11)
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

    keel::wavelet::FilterBank bad_fb;
    auto bad_dec2 = keel::wavelet::wavedec2(img, 10, 12, 2, bad_fb);
    check(bad_dec2.c.empty() && bad_dec2.s_rows.empty(), "wavedec2 invalid fb empty");
    auto bad_rec2 = keel::wavelet::waverec2(dec2.c, dec2.s_rows, dec2.s_cols, bad_fb);
    check(bad_rec2.rows == 0 && bad_rec2.cols == 0, "waverec2 invalid fb empty");

    // Haar orthogonal, integer lifting, and complex roundtrip
    const keel::f64 hx[8] = {3.0, 7.0, -2.0, 5.0, 11.0, 0.0, 4.0, -8.0};
    keel::f64 ha[4]{}, hd[4]{}, hrec[8]{};
    check(keel::wavelet::haar_max_level(8) == 3, "haar_max_level(8)");
    check(keel::wavelet::haar_max_level(0) == 0, "haar_max_level(0)");
    keel::wavelet::haar_step(hx, ha, hd, false);
    keel::wavelet::ihaar_step(ha, hd, hrec, false);
    for (keel::usize i = 0; i < 8; ++i) check_near(hrec[i], hx[i], 1e-14, "haar orth roundtrip");

    keel::wavelet::haar_step(hx, ha, hd, true);
    keel::wavelet::ihaar_step(ha, hd, hrec, true);
    for (keel::usize i = 0; i < 8; ++i) check_near(hrec[i], hx[i], 1e-15, "haar integer lifting roundtrip");

    const keel::c64 hxc[4] = {keel::c64(1.0, 2.0), keel::c64(-3.0, 4.0), keel::c64(5.0, -1.0), keel::c64(0.5, 2.5)};
    keel::c64 hac[2]{}, hdc[2]{}, hrecc[4]{};
    keel::wavelet::haar_step_c64(hxc, hac, hdc);
    keel::wavelet::ihaar_step_c64(hac, hdc, hrecc);
    for (keel::usize i = 0; i < 4; ++i) {
        check_near(hrecc[i].real(), hxc[i].real(), 1e-14, "haar_c64 roundtrip re");
        check_near(hrecc[i].imag(), hxc[i].imag(), 1e-14, "haar_c64 roundtrip im");
    }

    // Dyadic down/up including in-place aliasing regression
    keel::f64 dd[4]{}, du[9]{};
    keel::wavelet::dyaddown_1d(hx, dd, false);
    check_near(dd[0], 7.0, 1e-15, "dyaddown even 0");
    keel::wavelet::dyadup_1d(dd, du, false);
    check_near(du[0], 7.0, 1e-15, "dyadup 0");
    check_near(du[1], 0.0, 1e-15, "dyadup 1");
    check_near(du[2], 5.0, 1e-15, "dyadup 2");

    // In-place dyadup_1d (pad_ends = false and true)
    keel::f64 inplace_u[9] = {10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    keel::wavelet::dyadup_1d(keel::ArrayRef<keel::f64>(inplace_u, 4), keel::MutArrayRef<keel::f64>(inplace_u, 7),
                             false);
    check_near(inplace_u[0], 10.0, 1e-15, "dyadup inplace false [0]");
    check_near(inplace_u[1], 0.0, 1e-15, "dyadup inplace false [1]");
    check_near(inplace_u[2], 20.0, 1e-15, "dyadup inplace false [2]");
    check_near(inplace_u[6], 40.0, 1e-15, "dyadup inplace false [6]");

    keel::f64 inplace_p[9] = {10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    keel::wavelet::dyadup_1d(keel::ArrayRef<keel::f64>(inplace_p, 4), keel::MutArrayRef<keel::f64>(inplace_p, 9), true);
    check_near(inplace_p[0], 0.0, 1e-15, "dyadup inplace true [0]");
    check_near(inplace_p[1], 10.0, 1e-15, "dyadup inplace true [1]");
    check_near(inplace_p[3], 20.0, 1e-15, "dyadup inplace true [3]");
    check_near(inplace_p[7], 40.0, 1e-15, "dyadup inplace true [7]");
    check_near(inplace_p[8], 0.0, 1e-15, "dyadup inplace true [8]");
}

void test_shapes_centfrq_cwt() {
    keel::f64 x[11]{}, psi[11]{};
    keel::wavelet::mexihat(-5.0, 5.0, psi, x);
    check_near(x[5], 0.0, 1e-15, "mexihat center x");
    check(psi[5] > 0.8, "mexihat peak > 0.8");

    keel::wavelet::morlet(-4.0, 4.0, psi, x);
    check_near(psi[5], 1.0, 1e-15, "morlet(0) == 1");
    check_near(keel::wavelet::meyeraux(0.5), 0.5, 1e-15, "meyeraux(0.5) == 0.5");
    check(keel::is_nan(keel::wavelet::meyeraux(keel::kNaN)), "meyeraux(NaN) == NaN");

    keel::wavelet::gauswavf(-5.0, 5.0, 2, psi, x);
    check(keel::is_finite(psi[5]), "gauswavf finite");

    // cgauwavf including reversed bounds lb > ub regression
    keel::c64 cpsi[11]{}, cpsi_rev[11]{};
    keel::f64 x_rev[11]{};
    keel::wavelet::cgauwavf(-5.0, 5.0, 1, cpsi, x);
    keel::wavelet::cgauwavf(5.0, -5.0, 1, cpsi_rev, x_rev);
    check(keel::abs(cpsi[5]) > 0.0, "cgauwavf non-zero");
    check_near(cpsi_rev[5].imag(), cpsi[5].imag(), 1e-14, "cgauwavf reversed bounds preserves norm");

    // fbspwavf with odd m=3 across negative sinc lobes (integer power, never NaN)
    keel::c64 fbsp_out[41]{};
    keel::f64 fbsp_x[41]{};
    keel::wavelet::fbspwavf(-20.0, 20.0, 3, 1.0, 1.5, fbsp_out, fbsp_x);
    bool all_fbsp_finite = true;
    for (keel::usize i = 0; i < 41; ++i) {
        if (!keel::is_finite(fbsp_out[i].real()) || !keel::is_finite(fbsp_out[i].imag())) {
            all_fbsp_finite = false;
        }
    }
    check(all_fbsp_finite, "fbspwavf odd m=3 all finite");

    auto cf_morl = keel::wavelet::centfrq("morl", 8, true);
    check(cf_morl.ok, "centfrq morl ok");
    check_near(cf_morl.freq, 0.8125, 0.05, "centfrq morl ~0.8125");
    check(cf_morl.recfreq_real.size() == cf_morl.xval.size(), "centfrq recfreq size");

    auto cf_db4 = keel::wavelet::centfrq("db4", 8, false);
    check(cf_db4.ok && cf_db4.freq > 0.5 && cf_db4.freq < 1.0, "centfrq db4");
    auto cf_bad = keel::wavelet::centfrq("notawavelet", 8, false);
    check(!cf_bad.ok, "centfrq invalid name");

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

    // CWT invalid frequency limits rejected
    auto cwt_bad = keel::wavelet::cwt(tone, keel::wavelet::CwtWavelet::Morse, kFs, 10, 60.0, 3.0, true, 100.0, 50.0);
    check(cwt_bad.num_scales == 0, "cwt fmax <= fmin rejected");
    auto cwt_neg = keel::wavelet::cwt(tone, keel::wavelet::CwtWavelet::Morse, kFs, 10, 60.0, 3.0, true, -1.0, 50.0);
    check(cwt_neg.num_scales == 0, "cwt negative fmin rejected");
}

void test_wextend() {
    const keel::f64 x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto sym = keel::wavelet::wextend_1d(x, 2, keel::wavelet::ExtendSide::kBoth, keel::wavelet::ExtendMode::kSym);
    check(sym.size() == 9 && sym[0] == 2.0 && sym[1] == 1.0 && sym[7] == 5.0 && sym[8] == 4.0, "wextend sym");

    auto symw = keel::wavelet::wextend_1d(x, 2, keel::wavelet::ExtendSide::kBoth, keel::wavelet::ExtendMode::kSymW);
    check(symw.size() == 9 && symw[0] == 3.0 && symw[1] == 2.0 && symw[7] == 4.0 && symw[8] == 3.0, "wextend symw");

    auto sp1 = keel::wavelet::wextend_1d(x, 2, keel::wavelet::ExtendSide::kBoth, keel::wavelet::ExtendMode::kSp1);
    check(sp1.size() == 9 && sp1[0] == -1.0 && sp1[1] == 0.0 && sp1[7] == 6.0 && sp1[8] == 7.0, "wextend sp1");

    auto per = keel::wavelet::wextend_1d(x, 2, keel::wavelet::ExtendSide::kBoth, keel::wavelet::ExtendMode::kPer);
    check(per.size() == 10 && per[0] == 5.0 && per[1] == 5.0 && per[7] == 5.0 && per[8] == 1.0, "wextend per odd pad");

    // Regression: lf > n across ALL 9 modes on non-linear [1, 4, 2] with lf = 7
    const keel::f64 x3[3] = {1.0, 4.0, 2.0};
    auto asym_long =
        keel::wavelet::wextend_1d(x3, 7, keel::wavelet::ExtendSide::kBoth, keel::wavelet::ExtendMode::kAsym);
    const keel::f64 exp_asym[17] = {-1, 1, 4, 2, -2, -4, -1, 1, 4, 2, -2, -4, -1, 1, 4, 2, -2};
    check(asym_long.size() == 17, "wextend asym lf>n size");
    for (keel::usize i = 0; i < 17; ++i) check_near(asym_long[i], exp_asym[i], 1e-15, "wextend asym lf>n");

    auto asymw_long =
        keel::wavelet::wextend_1d(x3, 7, keel::wavelet::ExtendSide::kBoth, keel::wavelet::ExtendMode::kAsymW);
    const keel::f64 exp_asymw[17] = {0, -2, -4, -1, 2, 0, -2, 1, 4, 2, 0, 3, 6, 4, 2, 5, 8};
    check(asymw_long.size() == 17, "wextend asymw lf>n size");
    for (keel::usize i = 0; i < 17; ++i) check_near(asymw_long[i], exp_asymw[i], 1e-15, "wextend asymw lf>n");

    auto per_long = keel::wavelet::wextend_1d(x3, 7, keel::wavelet::ExtendSide::kBoth, keel::wavelet::ExtendMode::kPer);
    check(per_long.size() == 18 && per_long[0] == 4.0 && per_long[17] == 2.0, "wextend per lf>n");

    auto ppd_long = keel::wavelet::wextend_1d(x3, 7, keel::wavelet::ExtendSide::kBoth, keel::wavelet::ExtendMode::kPpd);
    check(ppd_long.size() == 17 && ppd_long[0] == 2.0 && ppd_long[16] == 1.0, "wextend ppd lf>n");

    // Regression: n == 1 and empty input across all 9 modes (no OOB/crash)
    const keel::f64 x1[1] = {5.0};
    const keel::wavelet::ExtendMode all_modes[9] = {
        keel::wavelet::ExtendMode::kSym,   keel::wavelet::ExtendMode::kSymW, keel::wavelet::ExtendMode::kAsym,
        keel::wavelet::ExtendMode::kAsymW, keel::wavelet::ExtendMode::kSp0,  keel::wavelet::ExtendMode::kSp1,
        keel::wavelet::ExtendMode::kPer,   keel::wavelet::ExtendMode::kZpd,  keel::wavelet::ExtendMode::kPpd,
    };
    for (keel::usize m = 0; m < 9; ++m) {
        auto y1 = keel::wavelet::wextend_1d(x1, 3, keel::wavelet::ExtendSide::kBoth, all_modes[m]);
        const keel::usize exp_len = (all_modes[m] == keel::wavelet::ExtendMode::kPer) ? 8 : 7;
        check(y1.size() == exp_len, "wextend n=1 length");
        auto y0 = keel::wavelet::wextend_1d(keel::ArrayRef<keel::f64>{}, 3, keel::wavelet::ExtendSide::kBoth,
                                            all_modes[m]);
        check(y0.empty(), "wextend empty returns empty");
    }

    const keel::c64 xc[3] = {keel::c64(1.0, 2.0), keel::c64(3.0, 4.0), keel::c64(5.0, 6.0)};
    auto zc = keel::wavelet::wextend_1d_c64(xc, 1, keel::wavelet::ExtendSide::kLeft, keel::wavelet::ExtendMode::kZpd);
    check(zc.size() == 4 && zc[0] == keel::c64(0.0, 0.0) && zc[1] == keel::c64(1.0, 2.0), "wextend c64 zpd left");
}

template <typename T>
bool write_bin(keel::StringRef dir, keel::StringRef name, const T* data, keel::usize count) {
    keel::String path(dir);
    keel::path_append(path, name);
    const keel::u8* bytes = reinterpret_cast<const keel::u8*>(data);
    const keel::Error err = keel::fs_write_file(path, keel::ArrayRef<keel::u8>(bytes, count * sizeof(T)));
    return err.is_ok();
}

keel::i32 dump_reference(keel::StringRef dir) {
    // 1. All 55 filter banks
    for (keel::usize idx = 0; idx < 55; ++idx) {
        keel::wavelet::FilterBank fb;
        if (!keel::wavelet::lookup_filters(kAll55Wavelets[idx], fb)) return 2;
        keel::String base_lo_d = keel::format("fb_%s_lod.bin", kAll55Wavelets[idx]);
        keel::String base_hi_d = keel::format("fb_%s_hid.bin", kAll55Wavelets[idx]);
        keel::String base_lo_r = keel::format("fb_%s_lor.bin", kAll55Wavelets[idx]);
        keel::String base_hi_r = keel::format("fb_%s_hir.bin", kAll55Wavelets[idx]);
        if (!write_bin(dir, base_lo_d, fb.lo_d.data(), fb.lo_d.size())) return 2;
        if (!write_bin(dir, base_hi_d, fb.hi_d.data(), fb.hi_d.size())) return 2;
        if (!write_bin(dir, base_lo_r, fb.lo_r.data(), fb.lo_r.size())) return 2;
        if (!write_bin(dir, base_hi_r, fb.hi_r.data(), fb.hi_r.size())) return 2;
    }

    // 2. Signals & 1D DWT/IDWT/wavedec/waverec/wrcoef/upcoef/wenergy/denoise
    keel::f64 sig16[16];
    for (keel::usize i = 0; i < 16; ++i) {
        sig16[i] = keel::sin(0.3 * keel::as<keel::f64>(i)) + 0.5 * keel::cos(0.7 * keel::as<keel::f64>(i));
    }
    if (!write_bin(dir, "sig16.bin", sig16, 16)) return 2;

    const keel::f64 sig3[3] = {1.0, 2.0, 3.0};
    if (!write_bin(dir, "sig3.bin", sig3, 3)) return 2;

    const char* dwt_waves[6] = {"haar", "db4", "sym4", "coif2", "bior2.4", "rbio4.4"};
    for (const char* wn : dwt_waves) {
        keel::wavelet::FilterBank fb;
        if (!keel::wavelet::lookup_filters(wn, fb)) return 2;
        auto d1 = keel::wavelet::dwt(sig16, fb.lo_d, fb.hi_d);
        auto r1 = keel::wavelet::idwt(d1.ca, d1.cd, fb.lo_r, fb.hi_r, 16);
        auto dec = keel::wavelet::wavedec(sig16, 3, fb);
        auto wrec = keel::wavelet::waverec(dec.c, dec.l, fb);
        if (!write_bin(dir, keel::format("dwt_%s_ca.bin", wn), d1.ca.data(), d1.ca.size())) return 2;
        if (!write_bin(dir, keel::format("dwt_%s_cd.bin", wn), d1.cd.data(), d1.cd.size())) return 2;
        if (!write_bin(dir, keel::format("idwt_%s.bin", wn), r1.data(), r1.size())) return 2;
        if (!write_bin(dir, keel::format("wavedec_%s_c.bin", wn), dec.c.data(), dec.c.size())) return 2;
        if (!write_bin(dir, keel::format("waverec_%s.bin", wn), wrec.data(), wrec.size())) return 2;
    }

    // Short signal n=3 with db4 + idwt([], cd)
    {
        keel::wavelet::FilterBank db4;
        if (!keel::wavelet::lookup_filters("db4", db4)) return 2;
        auto d3 = keel::wavelet::dwt(sig3, db4.lo_d, db4.hi_d);
        auto r3_d = keel::wavelet::idwt(keel::ArrayRef<keel::f64>{}, d3.cd, db4.lo_r, db4.hi_r);
        if (!write_bin(dir, "dwt_short_ca.bin", d3.ca.data(), d3.ca.size())) return 2;
        if (!write_bin(dir, "dwt_short_cd.bin", d3.cd.data(), d3.cd.size())) return 2;
        if (!write_bin(dir, "idwt_short_donly.bin", r3_d.data(), r3_d.size())) return 2;
    }

    // Multilevel utilities on db3
    {
        keel::wavelet::FilterBank db3;
        if (!keel::wavelet::lookup_filters("db3", db3)) return 2;
        auto dec = keel::wavelet::wavedec(sig16, 3, db3);
        keel::f64 ea = 0.0;
        keel::f64 ed[3]{};
        if (!keel::wavelet::wenergy(dec.c, dec.l, ea, ed)) return 2;
        const keel::f64 en[4] = {ea, ed[0], ed[1], ed[2]};
        if (!write_bin(dir, "wenergy_db3.bin", en, 4)) return 2;

        auto a2 = keel::wavelet::wrcoef(false, dec.c, dec.l, db3.lo_r, db3.hi_r, 2);
        auto d2 = keel::wavelet::wrcoef(true, dec.c, dec.l, db3.lo_r, db3.hi_r, 2);
        if (!write_bin(dir, "wrcoef_a2.bin", a2.data(), a2.size())) return 2;
        if (!write_bin(dir, "wrcoef_d2.bin", d2.data(), d2.size())) return 2;

        auto up_a = keel::wavelet::upcoef(false, sig3, db3, 2);
        auto up_d = keel::wavelet::upcoef(true, sig3, db3, 1);
        auto up_crop = keel::wavelet::upcoef(false, sig3, db3, 2, 10);
        if (!write_bin(dir, "upcoef_a2.bin", up_a.data(), up_a.size())) return 2;
        if (!write_bin(dir, "upcoef_d1.bin", up_d.data(), up_d.size())) return 2;
        if (!write_bin(dir, "upcoef_a2_crop10.bin", up_crop.data(), up_crop.size())) return 2;
    }

    // 3. 2D DWT / IDWT / wavedec2 / waverec2
    {
        keel::wavelet::FilterBank db2;
        if (!keel::wavelet::lookup_filters("db2", db2)) return 2;
        keel::f64 img[6 * 8];
        for (keel::usize i = 0; i < 48; ++i) img[i] = keel::as<keel::f64>(i % 7) - 3.0;
        if (!write_bin(dir, "img6x8.bin", img, 48)) return 2;
        auto d2 = keel::wavelet::dwt2(img, 6, 8, db2);
        auto r2 = keel::wavelet::idwt2(d2.ca, d2.ch, d2.cv, d2.cd, d2.out_rows, d2.out_cols, db2, 6, 8);
        if (!write_bin(dir, "dwt2_ca.bin", d2.ca.data(), d2.ca.size())) return 2;
        if (!write_bin(dir, "dwt2_ch.bin", d2.ch.data(), d2.ch.size())) return 2;
        if (!write_bin(dir, "dwt2_cv.bin", d2.cv.data(), d2.cv.size())) return 2;
        if (!write_bin(dir, "dwt2_cd.bin", d2.cd.data(), d2.cd.size())) return 2;
        if (!write_bin(dir, "idwt2_rec.bin", r2.out.data(), r2.out.size())) return 2;

        auto dec2 = keel::wavelet::wavedec2(img, 6, 8, 2, db2);
        auto rec2 = keel::wavelet::waverec2(dec2.c, dec2.s_rows, dec2.s_cols, db2);
        if (!write_bin(dir, "wavedec2_c.bin", dec2.c.data(), dec2.c.size())) return 2;
        if (!write_bin(dir, "waverec2_rec.bin", rec2.out.data(), rec2.out.size())) return 2;
    }

    // 4. SWT & MODWT & IMODWT
    {
        keel::wavelet::FilterBank db2;
        if (!keel::wavelet::lookup_filters("db2", db2)) return 2;
        auto sw = keel::wavelet::swt_ad(sig16, 2, db2.lo_d, db2.hi_d);
        if (!write_bin(dir, "swt_swa.bin", sw.swa_colmajor.data(), sw.swa_colmajor.size())) return 2;
        if (!write_bin(dir, "swt_swd.bin", sw.swd_colmajor.data(), sw.swd_colmajor.size())) return 2;

        // Circular shift invariance test for SWT: shift sig16 by 3 samples
        keel::f64 sig16_sh[16];
        for (keel::usize i = 0; i < 16; ++i) sig16_sh[i] = sig16[(i + 3) % 16];
        auto sw_sh = keel::wavelet::swt_ad(sig16_sh, 2, db2.lo_d, db2.hi_d);
        if (!write_bin(dir, "swt_swa_sh3.bin", sw_sh.swa_colmajor.data(), sw_sh.swa_colmajor.size())) return 2;
        if (!write_bin(dir, "swt_swd_sh3.bin", sw_sh.swd_colmajor.data(), sw_sh.swd_colmajor.size())) return 2;

        auto mw = keel::wavelet::modwt(sig16, 2, db2);
        auto imw = keel::wavelet::imodwt(mw, 2, 16, db2);
        if (!write_bin(dir, "modwt_db2.bin", mw.data(), mw.size())) return 2;
        if (!write_bin(dir, "imodwt_db2.bin", imw.data(), imw.size())) return 2;
    }

    // 5. Denoise / threshold / entropy
    {
        const keel::f64 v[4] = {-2.0, 0.0, 0.5, 3.0};
        keel::f64 t_soft[4]{}, t_hard[4]{};
        keel::wavelet::threshold_soft(v, t_soft, 1.0);
        keel::wavelet::threshold_hard(v, t_hard, 1.0);
        const keel::f64 c_mad[6] = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
        const keel::f64 mad = keel::wavelet::noise_sigma_mad(c_mad);
        const keel::f64 ents[5] = {
            keel::wavelet::entropy(v, keel::wavelet::EntropyKind::Shannon),
            keel::wavelet::entropy(v, keel::wavelet::EntropyKind::LogEnergy),
            keel::wavelet::entropy(v, keel::wavelet::EntropyKind::Threshold, 1.0),
            keel::wavelet::entropy(v, keel::wavelet::EntropyKind::Sure, 1.0),
            keel::wavelet::entropy(v, keel::wavelet::EntropyKind::Norm, 1.5),
        };
        if (!write_bin(dir, "thresh_soft.bin", t_soft, 4)) return 2;
        if (!write_bin(dir, "thresh_hard.bin", t_hard, 4)) return 2;
        if (!write_bin(dir, "noise_mad.bin", &mad, 1)) return 2;
        if (!write_bin(dir, "entropies.bin", ents, 5)) return 2;

        keel::wavelet::FilterBank db4;
        if (!keel::wavelet::lookup_filters("db4", db4)) return 2;
        auto dn = keel::wavelet::denoise_visushrink(sig16, 2, db4);
        if (!write_bin(dir, "denoise_visu.bin", dn.data(), dn.size())) return 2;
    }

    // 6. All 9 wextend modes on nonlinear [1, 4, 2] with lf=7 (lf > n)
    {
        const keel::f64 x3[3] = {1.0, 4.0, 2.0};
        const keel::wavelet::ExtendMode modes[9] = {
            keel::wavelet::ExtendMode::kSym,   keel::wavelet::ExtendMode::kSymW, keel::wavelet::ExtendMode::kAsym,
            keel::wavelet::ExtendMode::kAsymW, keel::wavelet::ExtendMode::kSp0,  keel::wavelet::ExtendMode::kSp1,
            keel::wavelet::ExtendMode::kPer,   keel::wavelet::ExtendMode::kZpd,  keel::wavelet::ExtendMode::kPpd,
        };
        const char* mnames[9] = {"sym", "symw", "asym", "asymw", "sp0", "sp1", "per", "zpd", "ppd"};
        for (keel::usize m = 0; m < 9; ++m) {
            auto y = keel::wavelet::wextend_1d(x3, 7, keel::wavelet::ExtendSide::kBoth, modes[m]);
            if (!write_bin(dir, keel::format("wext_%s.bin", mnames[m]), y.data(), y.size())) return 2;
        }
    }

    // 7. Continuous wavelet shapes, centfrq, wavefun_psi, and CWT
    {
        constexpr keel::usize kM = 101;
        keel::f64 gx[kM]{}, gp[kM]{};
        keel::c64 cp[kM]{};

        keel::wavelet::mexihat(-5.0, 5.0, gp, gx);
        if (!write_bin(dir, "shape_mexihat.bin", gp, kM)) return 2;

        keel::wavelet::morlet(-4.0, 4.0, gp, gx);
        if (!write_bin(dir, "shape_morlet.bin", gp, kM)) return 2;

        const keel::f64 mv_in[6] = {-0.5, 0.0, 0.25, 0.5, 1.0, 1.5};
        keel::f64 mv_out[6]{};
        keel::wavelet::meyeraux(mv_in, mv_out);
        if (!write_bin(dir, "shape_meyeraux.bin", mv_out, 6)) return 2;

        keel::wavelet::shanwavf(-20.0, 20.0, 1.0, 1.5, cp, gx);
        if (!write_bin(dir, "shape_shan.bin", cp, kM)) return 2;

        keel::wavelet::cmorwavf(-8.0, 8.0, 1.5, 1.0, cp, gx);
        if (!write_bin(dir, "shape_cmor.bin", cp, kM)) return 2;

        keel::wavelet::fbspwavf(-20.0, 20.0, 2, 1.0, 1.5, cp, gx);
        if (!write_bin(dir, "shape_fbsp.bin", cp, kM)) return 2;

        for (keel::i32 p = 1; p <= 8; ++p) {
            keel::wavelet::gauswavf(-5.0, 5.0, p, gp, gx);
            if (!write_bin(dir, keel::format("shape_gaus%d.bin", p), gp, kM)) return 2;
            keel::wavelet::cgauwavf(-5.0, 5.0, p, cp, gx);
            if (!write_bin(dir, keel::format("shape_cgau%d.bin", p), cp, kM)) return 2;
        }

        const char* cf_names[9] = {"morl", "mexh", "haar", "db4", "sym4", "coif2", "bior2.4", "cmor1.5-1", "cgau4"};
        keel::f64 cfs[9]{};
        for (keel::usize i = 0; i < 9; ++i) {
            auto r = keel::wavelet::centfrq(cf_names[i], 8, false);
            if (!r.ok) return 2;
            cfs[i] = r.freq;
        }
        if (!write_bin(dir, "centfrqs.bin", cfs, 9)) return 2;

        auto wf_db4 = keel::wavelet::wavefun_psi("db4", 6);
        if (!wf_db4.ok) return 2;
        if (!write_bin(dir, "wavefun_db4_psi.bin", wf_db4.psi_real.data(), wf_db4.psi_real.size())) return 2;

        constexpr keel::f64 kFs = 500.0;
        constexpr keel::usize kN = 256;
        keel::Array<keel::f64> tone(kN);
        for (keel::usize i = 0; i < kN; ++i) {
            tone[i] = keel::sin(keel::kTau * 50.0 * keel::as<keel::f64>(i) / kFs);
        }
        const keel::wavelet::CwtWavelet cwaves[3] = {
            keel::wavelet::CwtWavelet::Morse,
            keel::wavelet::CwtWavelet::Amor,
            keel::wavelet::CwtWavelet::Bump,
        };
        const char* cnames[3] = {"morse", "amor", "bump"};
        for (keel::usize w = 0; w < 3; ++w) {
            auto r = keel::wavelet::cwt(tone, cwaves[w], kFs, 10);
            if (!write_bin(dir, keel::format("cwt_%s_cfs.bin", cnames[w]), r.cfs_colmajor.data(),
                           r.cfs_colmajor.size()))
                return 2;
            if (!write_bin(dir, keel::format("cwt_%s_freqs.bin", cnames[w]), r.freqs.data(), r.freqs.size())) return 2;
            if (!write_bin(dir, keel::format("cwt_%s_coi.bin", cnames[w]), r.coi.data(), r.coi.size())) return 2;
        }
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && keel::StringRef(argv[1]) == "--dump-ref") {
        return dump_reference(keel::StringRef(argv[2]));
    }

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
    keel::eprintln("%d FAILURE(S)", g_failures);
    return 1;
}
