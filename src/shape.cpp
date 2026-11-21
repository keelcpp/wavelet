#include "keel/wavelet.h"

#include "keel/arena.h"
#include "keel/fft.h"
#include "keel/math.h"
#include "keel/number.h"
#include "keel/utf8.h"

namespace keel::wavelet {

namespace {

void fill_linspace(f64 lb, f64 ub, MutArrayRef<f64> x) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return;
    }
    if (n == 1) {
        x[0] = ub;
        return;
    }
    const f64 step = (ub - lb) / as<f64>(n - 1);
    for (usize i = 0; i < n; ++i) {
        x[i] = lb + step * as<f64>(i);
    }
    x[n - 1] = ub;
}

f64 sinc_pi(f64 x) noexcept {
    if (x == 0.0) {
        return 1.0;
    }
    const f64 a = kPi * x;
    return keel::sin(a) / a;
}

c64 cexp_2pi(f64 kx) noexcept {
    const f64 a = kTau * kx;
    return c64(keel::cos(a), keel::sin(a));
}

f64 hermite_real(i32 p, f64 t) noexcept {
    if (p <= 0) return 1.0;
    if (p == 1) return 2.0 * t;
    f64 hm = 1.0;
    f64 hc = 2.0 * t;
    for (i32 n = 1; n < p; ++n) {
        const f64 hn = 2.0 * t * hc - 2.0 * as<f64>(n) * hm;
        hm = hc;
        hc = hn;
    }
    return hc;
}

c64 hermite_complex(i32 p, c64 z) noexcept {
    if (p <= 0) return c64(1.0, 0.0);
    if (p == 1) return 2.0 * z;
    c64 hm(1.0, 0.0);
    c64 hc = 2.0 * z;
    for (i32 n = 1; n < p; ++n) {
        const c64 hn = 2.0 * z * hc - 2.0 * as<f64>(n) * hm;
        hm = hc;
        hc = hn;
    }
    return hc;
}

f64 dblfact_norm_sq(i32 p) noexcept {
    f64 r = keel::sqrt(kPi * 0.5);
    for (i32 k = 1; k <= p; ++k) {
        r *= as<f64>(2 * k - 1);
    }
    return r;
}

// Lanczos approximation for log(Gamma(z)), z > 0 (g = 7, 9 coefficients, ~1e-15 relative accuracy).
f64 lgamma_pos(f64 z) noexcept {
    static constexpr f64 kCoeffs[9] = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    };
    if (z < 0.5) {
        return keel::log(kPi / keel::sin(kPi * z)) - lgamma_pos(1.0 - z);
    }
    const f64 zm1 = z - 1.0;
    f64 x = kCoeffs[0];
    for (usize i = 1; i < 9; ++i) {
        x += kCoeffs[i] / (zm1 + as<f64>(i));
    }
    const f64 t = zm1 + 7.5;
    return 0.5 * keel::log(kTau) + (zm1 + 0.5) * keel::log(t) - t + keel::log(x);
}

Array<f64> conv_full_vec(ArrayRef<f64> a, ArrayRef<f64> b, Allocator* alloc) {
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

bool parse_two_f64(StringRef s, f64& v1, f64& v2) noexcept {
    const usize dash = s.find('-');
    if (dash == kInvalidIndex || dash == 0 || dash + 1 >= s.size()) {
        return false;
    }
    auto r1 = keel::parse_f64(s.substr(0, dash));
    auto r2 = keel::parse_f64(s.substr(dash + 1));
    if (!r1.is_ok() || !r2.is_ok()) {
        return false;
    }
    v1 = r1.value();
    v2 = r2.value();
    return (v1 > 0.0 && v2 > 0.0 && keel::is_finite(v1) && keel::is_finite(v2));
}

bool parse_i32_two_f64(StringRef s, i32& m, f64& v1, f64& v2) noexcept {
    const usize d1 = s.find('-');
    if (d1 == kInvalidIndex || d1 == 0 || d1 + 1 >= s.size()) {
        return false;
    }
    auto rm = keel::parse_i64(s.substr(0, d1));
    if (!rm.is_ok() || rm.value() < 1 || rm.value() > 1000) {
        return false;
    }
    m = as<i32>(rm.value());
    return parse_two_f64(s.substr(d1 + 1), v1, v2);
}

} // namespace

void mexihat(f64 lb, f64 ub, MutArrayRef<f64> psi, MutArrayRef<f64> x) noexcept {
    const usize n = (psi.size() < x.size()) ? psi.size() : x.size();
    if (n == 0) {
        return;
    }
    fill_linspace(lb, ub, MutArrayRef<f64>(x.data(), n));
    const f64 c = (2.0 / keel::sqrt(3.0)) * keel::pow(kPi, -0.25);
    for (usize i = 0; i < n; ++i) {
        const f64 t = x[i];
        const f64 t2 = t * t;
        psi[i] = c * (1.0 - t2) * keel::exp(-0.5 * t2);
    }
}

void morlet(f64 lb, f64 ub, MutArrayRef<f64> psi, MutArrayRef<f64> x) noexcept {
    const usize n = (psi.size() < x.size()) ? psi.size() : x.size();
    if (n == 0) {
        return;
    }
    fill_linspace(lb, ub, MutArrayRef<f64>(x.data(), n));
    for (usize i = 0; i < n; ++i) {
        const f64 t = x[i];
        psi[i] = keel::exp(-0.5 * t * t) * keel::cos(5.0 * t);
    }
}

f64 meyeraux(f64 v) noexcept {
    if (v <= 0.0) return 0.0;
    if (v >= 1.0) return 1.0;
    const f64 v2 = v * v;
    const f64 v3 = v2 * v;
    const f64 v4 = v2 * v2;
    return v4 * (35.0 - 84.0 * v + 70.0 * v2 - 20.0 * v3);
}

void meyeraux(ArrayRef<f64> in, MutArrayRef<f64> out) noexcept {
    const usize n = (in.size() < out.size()) ? in.size() : out.size();
    for (usize i = 0; i < n; ++i) {
        out[i] = meyeraux(in[i]);
    }
}

void shanwavf(f64 lb, f64 ub, f64 fb, f64 fc, MutArrayRef<c64> psi, MutArrayRef<f64> x) noexcept {
    const usize n = (psi.size() < x.size()) ? psi.size() : x.size();
    if (n == 0) {
        return;
    }
    fill_linspace(lb, ub, MutArrayRef<f64>(x.data(), n));
    const f64 s = keel::sqrt(fb);
    for (usize i = 0; i < n; ++i) {
        const f64 t = x[i];
        psi[i] = s * sinc_pi(fb * t) * cexp_2pi(fc * t);
    }
}

void cmorwavf(f64 lb, f64 ub, f64 fb, f64 fc, MutArrayRef<c64> psi, MutArrayRef<f64> x) noexcept {
    const usize n = (psi.size() < x.size()) ? psi.size() : x.size();
    if (n == 0) {
        return;
    }
    fill_linspace(lb, ub, MutArrayRef<f64>(x.data(), n));
    const f64 inv = 1.0 / keel::sqrt(kPi * fb);
    for (usize i = 0; i < n; ++i) {
        const f64 t = x[i];
        psi[i] = inv * keel::exp(-(t * t) / fb) * cexp_2pi(fc * t);
    }
}

void fbspwavf(f64 lb, f64 ub, i32 m, f64 fb, f64 fc, MutArrayRef<c64> psi, MutArrayRef<f64> x) noexcept {
    const usize n = (psi.size() < x.size()) ? psi.size() : x.size();
    if (n == 0 || m < 1) {
        return;
    }
    fill_linspace(lb, ub, MutArrayRef<f64>(x.data(), n));
    const f64 s = keel::sqrt(fb);
    const f64 inv_m = 1.0 / as<f64>(m);
    for (usize i = 0; i < n; ++i) {
        const f64 t = x[i];
        const f64 sc = sinc_pi(fb * t * inv_m);
        const f64 pow_sc = keel::pow(sc, as<f64>(m));
        psi[i] = s * pow_sc * cexp_2pi(fc * t);
    }
}

void gauswavf(f64 lb, f64 ub, i32 p, MutArrayRef<f64> psi, MutArrayRef<f64> x) noexcept {
    const usize n = (psi.size() < x.size()) ? psi.size() : x.size();
    if (n == 0 || p < 1) {
        return;
    }
    fill_linspace(lb, ub, MutArrayRef<f64>(x.data(), n));
    const f64 alpha_mag = 1.0 / keel::sqrt(dblfact_norm_sq(p));
    const f64 sgn = ((((p + 1) / 2) % 2) == 0) ? 1.0 : -1.0;
    const f64 scale = sgn * alpha_mag;
    for (usize i = 0; i < n; ++i) {
        const f64 t = x[i];
        psi[i] = scale * hermite_real(p, t) * keel::exp(-t * t);
    }
}

void cgauwavf(f64 lb, f64 ub, i32 p, MutArrayRef<c64> psi, MutArrayRef<f64> x, Allocator* alloc) {
    const usize n = (psi.size() < x.size()) ? psi.size() : x.size();
    if (n == 0 || p < 1) {
        return;
    }
    fill_linspace(lb, ub, MutArrayRef<f64>(x.data(), n));
    const f64 sgn_p = ((p % 2) == 0) ? 1.0 : -1.0;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> mag2(n, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        const f64 t = x[i];
        const c64 z(t, 0.5);
        const c64 hp = hermite_complex(p, z);
        const f64 e = keel::exp(-t * t);
        const c64 phase(keel::cos(t), -keel::sin(t));
        const c64 f = sgn_p * hp * (e * phase);
        psi[i] = f;
        mag2[i] = keel::norm(f);
    }

    f64 norm_sq = 0.0;
    if (n >= 2) {
        const f64 h = (ub - lb) / as<f64>(n - 1);
        for (usize i = 0; i < n; ++i) {
            const f64 w = (i == 0 || i + 1 == n) ? 0.5 : 1.0;
            norm_sq += w * mag2[i];
        }
        norm_sq *= h;
    } else {
        norm_sq = mag2[0];
    }
    const f64 inv_n = (norm_sq > 0.0) ? (1.0 / keel::sqrt(norm_sq)) : 0.0;
    for (usize i = 0; i < n; ++i) {
        psi[i] *= inv_n;
    }
}

WavefunPsiResult wavefun_psi(StringRef name, i32 iter, Allocator* alloc) {
    WavefunPsiResult res{Array<f64>(alloc), Array<c64>(alloc), Array<f64>(alloc), false, false};
    if (name.empty() || iter < 1 || iter > 20) {
        return res;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    String wname(name, scratch.allocator());
    for (usize i = 0; i < wname.size(); ++i) {
        wname[i] = keel::to_ascii_lower(wname[i]);
    }
    const StringRef wn = wname;
    const usize grid_n = as<usize>(1ULL << as<u32>(iter));

    if (wn == "morl") {
        res.psi_real.resize(grid_n);
        res.xval.resize(grid_n);
        morlet(-8.0, 8.0, res.psi_real, res.xval);
        res.ok = true;
        return res;
    }
    if (wn == "mexh") {
        res.psi_real.resize(grid_n);
        res.xval.resize(grid_n);
        mexihat(-8.0, 8.0, res.psi_real, res.xval);
        res.ok = true;
        return res;
    }
    if (wn.starts_with("gaus")) {
        i32 p = 1;
        if (wn.size() > 4) {
            auto rp = keel::parse_i64(wn.substr(4));
            if (!rp.is_ok() || rp.value() < 1 || rp.value() > 8) {
                return res;
            }
            p = as<i32>(rp.value());
        }
        res.psi_real.resize(grid_n);
        res.xval.resize(grid_n);
        gauswavf(-5.0, 5.0, p, res.psi_real, res.xval);
        res.ok = true;
        return res;
    }
    if (wn.starts_with("cgau")) {
        i32 p = 1;
        if (wn.size() > 4) {
            auto rp = keel::parse_i64(wn.substr(4));
            if (!rp.is_ok() || rp.value() < 1 || rp.value() > 8) {
                return res;
            }
            p = as<i32>(rp.value());
        }
        res.psi_cplx.resize(grid_n);
        res.xval.resize(grid_n);
        cgauwavf(-5.0, 5.0, p, res.psi_cplx, res.xval, scratch.allocator());
        res.is_complex = true;
        res.ok = true;
        return res;
    }
    if (wn.starts_with("shan")) {
        f64 fb = 1.0;
        f64 fc = 1.5;
        if (wn.size() > 4 && !parse_two_f64(wn.substr(4), fb, fc)) {
            return res;
        }
        res.psi_cplx.resize(grid_n);
        res.xval.resize(grid_n);
        shanwavf(-20.0, 20.0, fb, fc, res.psi_cplx, res.xval);
        res.is_complex = true;
        res.ok = true;
        return res;
    }
    if (wn.starts_with("cmor")) {
        f64 fb = 1.0;
        f64 fc = 1.5;
        if (wn.size() > 4 && !parse_two_f64(wn.substr(4), fb, fc)) {
            return res;
        }
        res.psi_cplx.resize(grid_n);
        res.xval.resize(grid_n);
        cmorwavf(-8.0, 8.0, fb, fc, res.psi_cplx, res.xval);
        res.is_complex = true;
        res.ok = true;
        return res;
    }
    if (wn.starts_with("fbsp")) {
        i32 m = 1;
        f64 fb = 1.0;
        f64 fc = 1.5;
        if (wn.size() > 4 && !parse_i32_two_f64(wn.substr(4), m, fb, fc)) {
            return res;
        }
        res.psi_cplx.resize(grid_n);
        res.xval.resize(grid_n);
        fbspwavf(-20.0, 20.0, m, fb, fc, res.psi_cplx, res.xval);
        res.is_complex = true;
        res.ok = true;
        return res;
    }

    FilterBank fb;
    if (!lookup_filters(wn, fb, scratch.allocator())) {
        return res;
    }
    const bool is_bior = (wn.starts_with("bior") || wn.starts_with("rbio"));
    Array<f64> lo(scratch.allocator());
    Array<f64> hi(scratch.allocator());
    if (is_bior) {
        lo.resize(fb.lo_d.size());
        for (usize i = 0; i < fb.lo_d.size(); ++i) {
            lo[i] = fb.lo_d[fb.lo_d.size() - 1 - i];
        }
        hi.resize(fb.hi_d.size());
        for (usize i = 0; i < fb.hi_d.size(); ++i) {
            hi[i] = fb.hi_d[fb.hi_d.size() - 1 - i];
        }
    } else {
        lo = move(fb.lo_r);
        hi = move(fb.hi_r);
    }

    const bool negate_psi = (wn.starts_with("sym") || wn.starts_with("coif") || wn.starts_with("bior2.") ||
                             wn.starts_with("bior3.") || wn == "bior4.4" || wn.starts_with("rbio2.") ||
                             wn.starts_with("rbio3.") || wn == "rbio4.4");
    if (negate_psi) {
        for (usize i = 0; i < hi.size(); ++i) {
            hi[i] = -hi[i];
        }
    }

    Array<f64> y = move(hi);
    for (i32 step = 1; step < iter; ++step) {
        Array<f64> up(2 * y.size() - 1, 0.0, scratch.allocator());
        for (usize k = 0; k < y.size(); ++k) {
            up[2 * k] = y[k];
        }
        y = conv_full_vec(ArrayRef<f64>(up.data(), up.size()), ArrayRef<f64>(lo.data(), lo.size()),
                          scratch.allocator());
    }
    const f64 scale = keel::pow(2.0, 0.5 * as<f64>(iter));
    for (usize i = 0; i < y.size(); ++i) {
        y[i] *= scale;
    }

    const usize l_len = lo.size();
    usize len_psi = 0;
    f64 t_end = 0.0;
    if (l_len == 2) {
        len_psi = grid_n + 2;
        t_end = 1.0 + 1.0 / as<f64>(grid_n);
    } else if (is_bior) {
        len_psi = (l_len - 1) * grid_n;
        t_end = as<f64>(l_len - 1) - 1.0 / as<f64>(grid_n);
    } else {
        len_psi = (l_len - 1) * grid_n + 1;
        t_end = as<f64>(l_len - 1);
    }

    res.psi_real.resize(len_psi);
    for (usize i = 0; i < len_psi; ++i) {
        res.psi_real[i] = 0.0;
    }
    const usize avail = (len_psi > 1) ? (len_psi - 1) : 0;
    const usize copy_len = (y.size() < avail) ? y.size() : avail;
    for (usize i = 0; i < copy_len; ++i) {
        res.psi_real[1 + i] = y[i];
    }

    res.xval.resize(len_psi);
    fill_linspace(0.0, t_end, res.xval);
    res.ok = true;
    return res;
}

CentfrqResult centfrq(StringRef name, i32 iter, bool with_recfreq, Allocator* alloc) {
    CentfrqResult res{0.0, Array<f64>(alloc), Array<f64>(alloc), Array<c64>(alloc), false, false};
    ArenaTemp scratch = keel::get_scratch(alloc);

    WavefunPsiResult wf = wavefun_psi(name, iter, scratch.allocator());
    if (!wf.ok || wf.xval.empty()) {
        return res;
    }
    const usize n = wf.xval.size();

    auto plan_res = fft::FftPlan1D::create(n, scratch.allocator());
    if (!plan_res.is_ok()) {
        return res;
    }
    const fft::FftPlan1D& plan = plan_res.value();

    Array<c64> spec(n, scratch.allocator());
    if (wf.is_complex) {
        plan.exec(ArrayRef<c64>(wf.psi_cplx.data(), n), spec, fft::FftDirection::Forward,
                  fft::FftNormalization::Backward);
    } else {
        plan.exec_real_to_complex(ArrayRef<f64>(wf.psi_real.data(), n), spec, fft::FftNormalization::Backward);
    }

    usize best_idx0 = 0;
    f64 best_mag = -1.0;
    for (usize i = 0; i < n; ++i) {
        const f64 mag = keel::hypot(spec[i].real(), spec[i].imag());
        if (mag > best_mag) {
            best_mag = mag;
            best_idx0 = i;
        }
    }

    const usize ind = best_idx0 + 1;
    const usize ind_fold = (as<f64>(ind) > 0.5 * as<f64>(n)) ? (n + 2 - ind) : ind;
    const f64 t_span = wf.xval[n - 1] - wf.xval[0];
    res.freq = as<f64>(ind_fold - 1) / t_span;
    res.is_complex = wf.is_complex;

    res.xval.resize(n);
    for (usize i = 0; i < n; ++i) {
        res.xval[i] = wf.xval[i];
    }

    if (with_recfreq) {
        const c64 y_peak = spec[best_idx0];
        const f64 inv_n = 1.0 / as<f64>(n);
        if (wf.is_complex) {
            res.recfreq_cplx.resize(n);
            for (usize k = 0; k < n; ++k) {
                const f64 ang = kTau * as<f64>(best_idx0 * k) * inv_n;
                const c64 e(keel::cos(ang), keel::sin(ang));
                res.recfreq_cplx[k] = (y_peak * e) * inv_n;
            }
        } else {
            res.recfreq_real.resize(n);
            const f64 scale =
                (best_idx0 == 0 || ((n & 1u) == 0u && best_idx0 == n / 2)) ? inv_n : (2.0 * inv_n);
            for (usize k = 0; k < n; ++k) {
                const f64 ang = kTau * as<f64>(best_idx0 * k) * inv_n;
                const c64 e(keel::cos(ang), keel::sin(ang));
                res.recfreq_real[k] = (y_peak * e).real() * scale;
            }
        }
    }

    res.ok = true;
    return res;
}

CwtResult cwt(ArrayRef<f64> x, CwtWavelet wavelet, f64 sample_rate, i32 voices_per_octave, f64 time_bandwidth,
              f64 gamma, bool extend_signal, f64 fmin, f64 fmax, Allocator* alloc) {
    CwtResult res{Array<c64>(alloc), Array<f64>(alloc), Array<f64>(alloc), 0, 0};
    const usize n = x.size();
    if (n < 4 || !(sample_rate > 0.0) || !keel::is_finite(sample_rate) || voices_per_octave < 1 ||
        voices_per_octave > 48) {
        return res;
    }
    if (wavelet == CwtWavelet::Morse) {
        if (!(gamma > 0.0) || !keel::is_finite(gamma) || !(time_bandwidth > gamma) ||
            time_bandwidth > 40.0 * gamma || !keel::is_finite(time_bandwidth)) {
            return res;
        }
    }

    const f64 nv = as<f64>(voices_per_octave);
    const f64 beta = time_bandwidth / gamma;

    f64 wpsi = 0.0;
    f64 wc = 0.0;
    f64 sigt = 0.0;

    if (wavelet == CwtWavelet::Morse) {
        wpsi = keel::pow(beta / gamma, 1.0 / gamma);
        const f64 target = keel::log(0.5);
        const f64 wpsi_ga = beta / gamma;
        f64 w = wpsi * (1.0 + keel::sqrt(2.0 * kLn2 / (beta * gamma)));
        for (i32 it = 0; it < 20; ++it) {
            const f64 w_ga = keel::pow(w, gamma);
            const f64 g = beta * keel::log(w / wpsi) - (w_ga - wpsi_ga) - target;
            const f64 gp = beta / w - gamma * (w_ga / w);
            if (keel::abs(gp) < 1e-30) {
                break;
            }
            const f64 step = g / gp;
            w -= step;
            if (w <= wpsi) {
                w = 0.5 * (w + step + wpsi);
            }
            if (keel::abs(step) <= 1e-15 * w) {
                break;
            }
        }
        wc = w;

        auto log_term = [&](f64 a) -> f64 {
            return lgamma_pos(a / gamma) - (a / gamma) * kLn2;
        };
        const f64 ld = log_term(2.0 * beta + 1.0);
        const f64 t1 = (beta * beta) * keel::exp(log_term(2.0 * beta - 1.0) - ld);
        const f64 t2 = (gamma * gamma) * keel::exp(log_term(2.0 * beta + 2.0 * gamma - 1.0) - ld);
        const f64 t3 = (2.0 * beta * gamma) * keel::exp(log_term(2.0 * beta + gamma - 1.0) - ld);
        const f64 inner = t1 + t2 - t3;
        sigt = wpsi * keel::sqrt(inner > 0.0 ? inner : 0.0);
    } else if (wavelet == CwtWavelet::Amor) {
        wpsi = 6.0;
        wc = wpsi + keel::sqrt(2.0 * kLn10);
        sigt = kSqrt2;
    } else {
        wpsi = 5.0;
        const f64 sig = 0.6;
        wc = wpsi + sig * keel::sqrt(1.0 - 1.0 / (1.0 - keel::log(0.1)));
        sigt = 5.847705277601328;
    }

    f64 s0 = wc / kPi;
    f64 smax = as<f64>(n) / (2.0 * sigt);
    const f64 s_min_max = s0 * keel::pow(2.0, 1.0 / nv);
    if (smax < s_min_max) {
        smax = s_min_max;
    }

    if (fmin > 0.0 && fmax > fmin && keel::is_finite(fmin) && keel::is_finite(fmax)) {
        const f64 fn_min = fmin / sample_rate;
        const f64 fn_max = fmax / sample_rate;
        const f64 s_from_fmax = wpsi / (kTau * fn_max);
        const f64 s_from_fmin = wpsi / (kTau * fn_min);
        if (s_from_fmax > s0) s0 = s_from_fmax;
        if (s_from_fmin < smax) smax = s_from_fmin;
        if (smax < s0) smax = s0;
    }

    f64 n_oct = keel::log2(smax / s0);
    if (n_oct < 0.0) n_oct = 0.0;
    const usize n_scales = 1 + as<usize>(keel::floor(nv * n_oct + 1e-12));

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> scales(n_scales, scratch.allocator());
    res.freqs.resize(n_scales);
    for (usize j = 0; j < n_scales; ++j) {
        scales[j] = s0 * keel::pow(2.0, as<f64>(j) / nv);
        res.freqs[j] = (wpsi * sample_rate) / (kTau * scales[j]);
    }

    const usize half_n = n / 2;
    const usize pad_l = extend_signal ? ((half_n < 100000) ? half_n : 100000) : 0;
    const usize n_pad = n + 2 * pad_l;

    Array<f64> x_pad(n_pad, scratch.allocator());
    for (usize i = 0; i < pad_l; ++i) {
        x_pad[i] = x[pad_l - 1 - i];
    }
    for (usize i = 0; i < n; ++i) {
        x_pad[pad_l + i] = x[i];
    }
    for (usize i = 0; i < pad_l; ++i) {
        x_pad[pad_l + n + i] = x[n - 1 - i];
    }

    auto plan_res = fft::FftPlan1D::create(n_pad, scratch.allocator());
    if (!plan_res.is_ok()) {
        return CwtResult{Array<c64>(alloc), Array<f64>(alloc), Array<f64>(alloc), 0, 0};
    }
    const fft::FftPlan1D& plan = plan_res.value();

    const usize k_max = n_pad / 2;
    Array<c64> x_half(k_max + 1, scratch.allocator());
    plan.exec_rfft(x_pad, x_half);

    Array<f64> omega(k_max + 1, scratch.allocator());
    const f64 d_omega = kTau / as<f64>(n_pad);
    for (usize k = 0; k <= k_max; ++k) {
        omega[k] = as<f64>(k) * d_omega;
    }

    res.cfs_colmajor.resize(n_scales * n);
    res.num_scales = n_scales;
    res.num_samples = n;

    Array<c64> prod_row(n_pad, scratch.allocator());
    Array<c64> y_row(n_pad, scratch.allocator());
    const f64 wpsi_ga = (wavelet == CwtWavelet::Morse) ? (beta / gamma) : 0.0;

    for (usize j = 0; j < n_scales; ++j) {
        const f64 s = scales[j];
        for (usize k = 0; k < n_pad; ++k) {
            prod_row[k] = c64(0.0, 0.0);
        }
        for (usize k = 1; k <= k_max; ++k) {
            const f64 sw = s * omega[k];
            f64 h_val = 0.0;
            if (wavelet == CwtWavelet::Morse) {
                if (sw > 0.0) {
                    const f64 log_h = beta * keel::log(sw / wpsi) - (keel::pow(sw, gamma) - wpsi_ga);
                    if (log_h > -745.0) {
                        h_val = 2.0 * keel::exp(log_h);
                    }
                }
            } else if (wavelet == CwtWavelet::Amor) {
                const f64 diff = sw - wpsi;
                h_val = 2.0 * keel::exp(-0.5 * diff * diff);
            } else {
                const f64 u = (sw - wpsi) / 0.6;
                if (keel::abs(u) < 1.0 - 1e-15) {
                    h_val = 2.0 * keel::exp(1.0 - 1.0 / (1.0 - u * u));
                }
            }
            if (h_val != 0.0) {
                prod_row[k] = x_half[k] * h_val;
            }
        }

        plan.exec(prod_row, y_row, fft::FftDirection::Inverse, fft::FftNormalization::Backward);
        for (usize i = 0; i < n; ++i) {
            res.cfs_colmajor[j + i * n_scales] = y_row[pad_l + i];
        }
    }

    res.coi.resize(n);
    const f64 coi_num = (sample_rate * wpsi * sigt) / kTau;
    const f64 f_top = res.freqs[0];
    for (usize i = 0; i < n; ++i) {
        const usize d_left = i + 1;
        const usize d_right = n - i;
        const usize dist = (d_left < d_right) ? d_left : d_right;
        const f64 val = coi_num / as<f64>(dist);
        res.coi[i] = (val < f_top) ? val : f_top;
    }

    return res;
}

} // namespace keel::wavelet
