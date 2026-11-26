#!/usr/bin/env python3
"""Differential reference verifier for keel::wavelet against MATLAB R2025b Wavelet Toolbox,
optional PyWavelets, and exact mathematical identities."""

import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ALL_55_WAVELETS = [
    "haar", "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10",
    "sym2", "sym3", "sym4", "sym5", "sym6", "sym7", "sym8", "sym9", "sym10",
    "coif1", "coif2", "coif3", "coif4", "coif5",
    "bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8",
    "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9", "bior4.4", "bior5.5", "bior6.8",
    "rbio1.1", "rbio1.3", "rbio1.5", "rbio2.2", "rbio2.4", "rbio2.6", "rbio2.8",
    "rbio3.1", "rbio3.3", "rbio3.5", "rbio3.7", "rbio3.9", "rbio4.4", "rbio5.5", "rbio6.8",
]

ORTH_WAVELETS = ALL_55_WAVELETS[:25]
BIOR_WAVELETS = ALL_55_WAVELETS[25:]
DWT_WAVES = ["haar", "db4", "sym4", "coif2", "bior2.4", "rbio4.4"]
EXT_MODES = ["sym", "symw", "asym", "asymw", "sp0", "sp1", "per", "zpd", "ppd"]
CENTFRQ_WAVES = ["morl", "mexh", "haar", "db4", "sym4", "coif2", "bior2.4", "cmor1.5-1", "cgau4"]


def load_f64(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


def load_c64(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<c16")


def record(name: str, got: np.ndarray, ref: np.ndarray, tol: float, results: list):
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    err = float(np.max(np.abs(got - ref))) if got.size > 0 else 0.0
    scale = max(1.0, float(np.max(np.abs(ref)))) if ref.size > 0 else 1.0
    rel = err / scale
    if not math.isfinite(rel) or rel > tol:
        raise AssertionError(f"{name}: max_abs={err:.3e}, rel={rel:.3e} > tol={tol:.3e}")
    results.append((name, rel))


def verify_mathematical_identities(tmp: str, results: list):
    # 1. All 55 filter banks: sum, norm, orthogonality / biorthogonality, QMF identities
    max_orth_rel = 0.0
    max_bior_rel = 0.0
    for wn in ALL_55_WAVELETS:
        lo_d = load_f64(os.path.join(tmp, f"fb_{wn}_lod.bin"))
        hi_d = load_f64(os.path.join(tmp, f"fb_{wn}_hid.bin"))
        lo_r = load_f64(os.path.join(tmp, f"fb_{wn}_lor.bin"))
        hi_r = load_f64(os.path.join(tmp, f"fb_{wn}_hir.bin"))

        # Low-pass sums = sqrt(2), high-pass sums = 0
        e_sum = max(
            abs(np.sum(lo_d) - math.sqrt(2.0)),
            abs(np.sum(lo_r) - math.sqrt(2.0)),
            abs(np.sum(hi_d)),
            abs(np.sum(hi_r)),
        )
        # Perfect reconstruction / no-aliasing: conv(lo_d, lo_r) + conv(hi_d, hi_r) = 2 * delta_{N-1}
        pr = np.convolve(lo_d, lo_r) + np.convolve(hi_d, hi_r)
        delta = np.zeros_like(pr)
        delta[len(lo_d) - 1] = 2.0
        e_pr = float(np.max(np.abs(pr - delta)))
        # Alternating flip alias cancellation: conv(lo_d * (-1)^k, lo_r) + conv(hi_d * (-1)^k, hi_r) = 0
        alt = (-1.0) ** np.arange(len(lo_d))
        alias = np.convolve(lo_d * alt, lo_r) + np.convolve(hi_d * alt, hi_r)
        e_alias = float(np.max(np.abs(alias)))

        if wn in ORTH_WAVELETS:
            # Orthogonality: ||lo_d||_2 == 1, lo_r == rev(lo_d), double-shift orthogonality
            e_norm = abs(float(np.linalg.norm(lo_d)) - 1.0)
            e_rev = float(np.max(np.abs(lo_r - lo_d[::-1])))
            ac = np.convolve(lo_d, lo_d[::-1])
            even_idx = np.arange(len(lo_d) - 1, -1, -2)
            even_idx = np.unique(np.concatenate([even_idx, np.arange(len(lo_d) - 1, len(ac), 2)]))
            ac_ref = np.zeros_like(ac)
            ac_ref[len(lo_d) - 1] = 1.0
            e_shift = float(np.max(np.abs(ac[even_idx] - ac_ref[even_idx])))
            err_w = max(e_sum, e_pr, e_alias, e_norm, e_rev, e_shift)
            if wn == "coif5":
                # Published Beylkin/Daubechies coif5 table has ~8.3e-9 intrinsic PR residual in MATLAB as well.
                assert err_w < 1e-8, f"coif5 filter bank identity error {err_w}"
                results.append(("math:orth_coif5_identities", err_w))
            else:
                max_orth_rel = max(max_orth_rel, err_w)
        else:
            max_bior_rel = max(max_bior_rel, e_sum, e_pr, e_alias)

    assert max_orth_rel < 1e-10, f"orth filter bank identity error {max_orth_rel}"
    assert max_bior_rel < 1e-11, f"bior filter bank identity error {max_bior_rel}"
    results.append(("math:orth_24_filters_identities", max_orth_rel))
    results.append(("math:bior_30_filters_identities", max_bior_rel))

    # 2. Perfect reconstruction for DWT/IDWT, wavedec/waverec, dwt2/idwt2, wavedec2/waverec2, modwt/imodwt
    sig16 = load_f64(os.path.join(tmp, "sig16.bin"))
    for wn in DWT_WAVES:
        record(f"math:idwt_pr_{wn}", load_f64(os.path.join(tmp, f"idwt_{wn}.bin")), sig16, 1e-11, results)
        record(f"math:waverec_pr_{wn}", load_f64(os.path.join(tmp, f"waverec_{wn}.bin")), sig16, 1e-10, results)

    img6x8 = load_f64(os.path.join(tmp, "img6x8.bin"))
    record("math:idwt2_pr_db2", load_f64(os.path.join(tmp, "idwt2_rec.bin")), img6x8, 1e-11, results)
    record("math:waverec2_pr_db2", load_f64(os.path.join(tmp, "waverec2_rec.bin")), img6x8, 1e-10, results)

    # 3. SWT shift-invariance (shifting input by 3 circularly shifts swa/swd by 3)
    swa = load_f64(os.path.join(tmp, "swt_swa.bin")).reshape((2, 16), order="F")
    swd = load_f64(os.path.join(tmp, "swt_swd.bin")).reshape((2, 16), order="F")
    swa_sh = load_f64(os.path.join(tmp, "swt_swa_sh3.bin")).reshape((2, 16), order="F")
    swd_sh = load_f64(os.path.join(tmp, "swt_swd_sh3.bin")).reshape((2, 16), order="F")
    record("math:swt_shift_invariance_swa", swa_sh, np.roll(swa, -3, axis=1), 1e-13, results)
    record("math:swt_shift_invariance_swd", swd_sh, np.roll(swd, -3, axis=1), 1e-13, results)

    # 4. MODWT reconstruction & energy conservation
    mw = load_f64(os.path.join(tmp, "modwt_db2.bin")).reshape((3, 16), order="F")
    imw = load_f64(os.path.join(tmp, "imodwt_db2.bin"))
    record("math:imodwt_pr_db2", imw, sig16, 1e-11, results)
    record("math:modwt_energy", np.array([np.sum(mw ** 2)]), np.array([np.sum(sig16 ** 2)]), 1e-12, results)

    # 5. Denoising threshold & entropy formulas
    record("math:thresh_soft", load_f64(os.path.join(tmp, "thresh_soft.bin")), np.array([-1.0, 0.0, 0.0, 2.0]), 1e-15, results)
    record("math:thresh_hard", load_f64(os.path.join(tmp, "thresh_hard.bin")), np.array([-2.0, 0.0, 0.0, 3.0]), 1e-15, results)
    record("math:noise_mad", load_f64(os.path.join(tmp, "noise_mad.bin")), np.array([3.5 / 0.6745]), 1e-14, results)
    v = np.array([-2.0, 0.0, 0.5, 3.0])
    v_nz = v[v != 0.0]
    ref_ents = np.array([
        -np.sum((v_nz ** 2) * np.log(v_nz ** 2)),
        np.sum(np.log(v_nz ** 2)),
        float(np.sum(np.abs(v) > 1.0)),
        len(v) - 2.0 * np.sum(np.abs(v) <= 1.0) + np.sum(np.minimum(v ** 2, 1.0)),
        np.sum(np.abs(v) ** 1.5),
    ])
    record("math:entropies", load_f64(os.path.join(tmp, "entropies.bin")), ref_ents, 1e-13, results)

    # 6. Analytic continuous wavelet shapes (exact closed-form formulas)
    t_mexh = np.linspace(-5.0, 5.0, 101)
    ref_mexh = (2.0 / math.sqrt(3.0)) * (math.pi ** -0.25) * (1.0 - t_mexh ** 2) * np.exp(-0.5 * t_mexh ** 2)
    record("math:mexihat", load_f64(os.path.join(tmp, "shape_mexihat.bin")), ref_mexh, 1e-14, results)

    t_morl = np.linspace(-4.0, 4.0, 101)
    ref_morl = np.exp(-0.5 * t_morl ** 2) * np.cos(5.0 * t_morl)
    record("math:morlet", load_f64(os.path.join(tmp, "shape_morlet.bin")), ref_morl, 1e-14, results)

    t_shan = np.linspace(-20.0, 20.0, 101)
    ref_shan = np.sinc(1.0 * t_shan) * np.exp(1j * 2.0 * math.pi * 1.5 * t_shan)
    record("math:shanwavf", load_c64(os.path.join(tmp, "shape_shan.bin")), ref_shan, 1e-13, results)

    t_cmor = np.linspace(-8.0, 8.0, 101)
    ref_cmor = (1.0 / math.sqrt(math.pi * 1.5)) * np.exp(-(t_cmor ** 2) / 1.5) * np.exp(1j * 2.0 * math.pi * 1.0 * t_cmor)
    record("math:cmorwavf", load_c64(os.path.join(tmp, "shape_cmor.bin")), ref_cmor, 1e-14, results)

    t_fbsp = np.linspace(-20.0, 20.0, 101)
    ref_fbsp = (np.sinc(t_fbsp / 2.0) ** 2) * np.exp(1j * 2.0 * math.pi * 1.5 * t_fbsp)
    record("math:fbspwavf", load_c64(os.path.join(tmp, "shape_fbsp.bin")), ref_fbsp, 1e-13, results)


def verify_pywt(tmp: str, results: list):
    try:
        import pywt
    except ImportError:
        return

    sig16 = load_f64(os.path.join(tmp, "sig16.bin"))
    for wn in DWT_WAVES:
        ca, cd = pywt.dwt(sig16, wn, mode="symmetric")
        record(f"pywt:dwt_{wn}_ca", load_f64(os.path.join(tmp, f"dwt_{wn}_ca.bin")), ca, 1e-11, results)
        record(f"pywt:dwt_{wn}_cd", load_f64(os.path.join(tmp, f"dwt_{wn}_cd.bin")), cd, 1e-11, results)


def verify_matlab(tmp: str, results: list):
    matlab_exe = shutil.which("matlab")
    if not matlab_exe:
        cand = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
        if os.path.exists(cand):
            matlab_exe = cand
    if not matlab_exe:
        print("NOTE: MATLAB not found; skipping MATLAB Wavelet Toolbox reference check.")
        return

    d = tmp.replace("\\", "/")
    m_script = os.path.join(tmp, "run_ref.m")
    # Write a single batched MATLAB script that emits all reference binaries in one invocation.
    all55_cell = "{" + ",".join(f"'{w}'" for w in ALL_55_WAVELETS) + "}"
    dwt_cell = "{" + ",".join(f"'{w}'" for w in DWT_WAVES) + "}"
    ext_cell = "{" + ",".join(f"'{m}'" for m in EXT_MODES) + "}"
    cf_cell = "{" + ",".join(f"'{w}'" for w in CENTFRQ_WAVES) + "}"

    m_code = f"""
d = '{d}';
wb = @(name, arr) fwrite_helper([d '/' name], arr);

% 1. All 55 filter banks
ws = {all55_cell};
for i = 1:numel(ws)
    w = ws{{i}};
    [ld, hd, lr, hr] = wfilters(w);
    wb(['m_fb_' w '_lod.bin'], ld);
    wb(['m_fb_' w '_hid.bin'], hd);
    wb(['m_fb_' w '_lor.bin'], lr);
    wb(['m_fb_' w '_hir.bin'], hr);
end

% 2. 1D DWT / wavedec / wenergy / wrcoef / upcoef
f = fopen([d '/sig16.bin'], 'rb'); sig16 = fread(f, 'double').'; fclose(f);
sig3 = [1, 2, 3];
dw = {dwt_cell};
for i = 1:numel(dw)
    w = dw{{i}};
    [ca, cd] = dwt(sig16, w);
    [c, l] = wavedec(sig16, 3, w);
    wb(['m_dwt_' w '_ca.bin'], ca);
    wb(['m_dwt_' w '_cd.bin'], cd);
    wb(['m_wavedec_' w '_c.bin'], c);
end

[ca3, cd3] = dwt(sig3, 'db4');
wb('m_dwt_short_ca.bin', ca3);
wb('m_dwt_short_cd.bin', cd3);
wb('m_idwt_short_donly.bin', idwt([], cd3, 'db4'));

[c3, l3] = wavedec(sig16, 3, 'db3');
[ea, ed] = wenergy(c3, l3);
wb('m_wenergy_db3.bin', [ea, ed]);
wb('m_wrcoef_a2.bin', wrcoef('a', c3, l3, 'db3', 2));
wb('m_wrcoef_d2.bin', wrcoef('d', c3, l3, 'db3', 2));
wb('m_upcoef_a2.bin', upcoef('a', sig3, 'db3', 2));
wb('m_upcoef_d1.bin', upcoef('d', sig3, 'db3', 1));
wb('m_upcoef_a2_crop10.bin', upcoef('a', sig3, 'db3', 2, 10));

% 3. 2D DWT / wavedec2
f = fopen([d '/img6x8.bin'], 'rb'); img = reshape(fread(f, 'double'), 6, 8); fclose(f);
[ca2, ch2, cv2, cd2] = dwt2(img, 'db2');
wb('m_dwt2_ca.bin', ca2(:));
wb('m_dwt2_ch.bin', ch2(:));
wb('m_dwt2_cv.bin', cv2(:));
wb('m_dwt2_cd.bin', cd2(:));
[c2d, ~] = wavedec2(img, 2, 'db2');
wb('m_wavedec2_c.bin', c2d(:));

% 4. SWT & MODWT
[swa, swd] = swt(sig16, 2, 'db2');
wb('m_swt_swa.bin', swa(:));
wb('m_swt_swd.bin', swd(:));
mw = modwt(sig16, 'db2', 2);
wb('m_modwt_db2.bin', mw(:));

% 5. Threshold / noise / entropy
v = [-2.0, 0.0, 0.5, 3.0];
wb('m_thresh_soft.bin', wthresh(v, 's', 1.0));
wb('m_thresh_hard.bin', wthresh(v, 'h', 1.0));
wb('m_noise_mad.bin', wnoisest([1, -2, 3, -4, 5, -6]));
wb('m_entropies.bin', [wentropy(v,'shannon'), wentropy(v,'log energy'), wentropy(v,'threshold',1.0), wentropy(v,'sure',1.0), wentropy(v,'norm',1.5)]);

% 6. All 9 wextend modes (n=3, lf=7)
x3 = [1, 4, 2];
em = {ext_cell};
for i = 1:numel(em)
    m = em{{i}};
    wb(['m_wext_' m '.bin'], wextend('1D', m, x3, 7));
end

% 7. Shapes, centfrq, wavefun, CWT
[p, ~] = mexihat(-5, 5, 101); wb('m_shape_mexihat.bin', p);
[p, ~] = morlet(-4, 4, 101); wb('m_shape_morlet.bin', p);
wb('m_shape_meyeraux.bin', meyeraux([-0.5, 0, 0.25, 0.5, 1.0, 1.5]));
[p, ~] = shanwavf(-20, 20, 101, 1.0, 1.5); wb_c([d '/m_shape_shan.bin'], p);
[p, ~] = cmorwavf(-8, 8, 101, 1.5, 1.0); wb_c([d '/m_shape_cmor.bin'], p);
[p, ~] = fbspwavf(-20, 20, 101, 2, 1.0, 1.5); wb_c([d '/m_shape_fbsp.bin'], p);
for ord = 1:8
    [gp, ~] = gauswavf(-5, 5, 101, ord); wb(sprintf('m_shape_gaus%d.bin', ord), gp);
    [cp, ~] = cgauwavf(-5, 5, 101, ord); wb_c(sprintf('%s/m_shape_cgau%d.bin', d, ord), cp);
end

cfw = {cf_cell};
cfs = zeros(1, numel(cfw));
for i = 1:numel(cfw)
    cfs(i) = centfrq(cfw{{i}}, 8);
end
wb('m_centfrqs.bin', cfs);
[~, psi_db4, ~] = wavefun('db4', 6);
wb('m_wavefun_db4_psi.bin', psi_db4);

fs = 500;
tone = sin(2*pi*50*(0:255)/fs);
cw = {{'morse', 'amor', 'bump'}};
for i = 1:3
    [wt, frq, coi] = cwt(tone, cw{{i}}, fs, 'VoicesPerOctave', 10);
    wb_c([d '/m_cwt_' cw{{i}} '_cfs.bin'], wt(:));
    wb(['m_cwt_' cw{{i}} '_freqs.bin'], frq(:));
    wb(['m_cwt_' cw{{i}} '_coi.bin'], coi(:));
end

function fwrite_helper(path, arr)
    f = fopen(path, 'wb');
    fwrite(f, double(arr(:)), 'double');
    fclose(f);
end

function wb_c(path, arr)
    arr = arr(:);
    inter = zeros(2 * numel(arr), 1);
    inter(1:2:end) = real(arr);
    inter(2:2:end) = imag(arr);
    f = fopen(path, 'wb');
    fwrite(f, double(inter), 'double');
    fclose(f);
end
"""
    with open(m_script, "w", encoding="utf-8") as fh:
        fh.write(m_code)

    proc = subprocess.run(
        [matlab_exe, "-batch", f"run('{m_script.replace(chr(92), '/')}')"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"MATLAB reference script failed:\n{proc.stdout}\n{proc.stderr}")

    # Compare all 55 filter banks against MATLAB wfilters
    max_fb_rel = 0.0
    for wn in ALL_55_WAVELETS:
        for suf in ("lod", "hid", "lor", "hir"):
            got = load_f64(os.path.join(tmp, f"fb_{wn}_{suf}.bin"))
            ref = load_f64(os.path.join(tmp, f"m_fb_{wn}_{suf}.bin"))
            assert got.shape == ref.shape, f"matlab:fb_{wn}_{suf} shape mismatch"
            rel = float(np.max(np.abs(got - ref))) / max(1.0, float(np.max(np.abs(ref))))
            if rel > 1e-11:
                raise AssertionError(f"matlab:fb_{wn}_{suf}: rel={rel:.3e}")
            max_fb_rel = max(max_fb_rel, rel)
    results.append(("matlab:all_55_wfilters", max_fb_rel))

    # 1D DWT & wavedec
    for wn in DWT_WAVES:
        record(f"matlab:dwt_{wn}_ca", load_f64(os.path.join(tmp, f"dwt_{wn}_ca.bin")),
               load_f64(os.path.join(tmp, f"m_dwt_{wn}_ca.bin")), 1e-11, results)
        record(f"matlab:dwt_{wn}_cd", load_f64(os.path.join(tmp, f"dwt_{wn}_cd.bin")),
               load_f64(os.path.join(tmp, f"m_dwt_{wn}_cd.bin")), 1e-11, results)
        record(f"matlab:wavedec_{wn}", load_f64(os.path.join(tmp, f"wavedec_{wn}_c.bin")),
               load_f64(os.path.join(tmp, f"m_wavedec_{wn}_c.bin")), 1e-10, results)

    # Short signal n=3 < lf with db4 & single-branch idwt([], cd)
    record("matlab:dwt_short_n3_ca", load_f64(os.path.join(tmp, "dwt_short_ca.bin")),
           load_f64(os.path.join(tmp, "m_dwt_short_ca.bin")), 1e-11, results)
    record("matlab:dwt_short_n3_cd", load_f64(os.path.join(tmp, "dwt_short_cd.bin")),
           load_f64(os.path.join(tmp, "m_dwt_short_cd.bin")), 1e-11, results)
    record("matlab:idwt_short_donly", load_f64(os.path.join(tmp, "idwt_short_donly.bin")),
           load_f64(os.path.join(tmp, "m_idwt_short_donly.bin")), 1e-11, results)

    # Multilevel utilities
    record("matlab:wenergy_db3", load_f64(os.path.join(tmp, "wenergy_db3.bin")),
           load_f64(os.path.join(tmp, "m_wenergy_db3.bin")), 1e-11, results)
    record("matlab:wrcoef_a2", load_f64(os.path.join(tmp, "wrcoef_a2.bin")),
           load_f64(os.path.join(tmp, "m_wrcoef_a2.bin")), 1e-10, results)
    record("matlab:wrcoef_d2", load_f64(os.path.join(tmp, "wrcoef_d2.bin")),
           load_f64(os.path.join(tmp, "m_wrcoef_d2.bin")), 1e-10, results)
    record("matlab:upcoef_a2", load_f64(os.path.join(tmp, "upcoef_a2.bin")),
           load_f64(os.path.join(tmp, "m_upcoef_a2.bin")), 1e-11, results)
    record("matlab:upcoef_d1", load_f64(os.path.join(tmp, "upcoef_d1.bin")),
           load_f64(os.path.join(tmp, "m_upcoef_d1.bin")), 1e-11, results)
    record("matlab:upcoef_a2_crop10", load_f64(os.path.join(tmp, "upcoef_a2_crop10.bin")),
           load_f64(os.path.join(tmp, "m_upcoef_a2_crop10.bin")), 1e-11, results)

    # 2D DWT & wavedec2
    for b in ("ca", "ch", "cv", "cd"):
        record(f"matlab:dwt2_{b}", load_f64(os.path.join(tmp, f"dwt2_{b}.bin")),
               load_f64(os.path.join(tmp, f"m_dwt2_{b}.bin")), 1e-11, results)
    record("matlab:wavedec2_db2", load_f64(os.path.join(tmp, "wavedec2_c.bin")),
           load_f64(os.path.join(tmp, "m_wavedec2_c.bin")), 1e-10, results)

    # SWT & MODWT
    record("matlab:swt_swa", load_f64(os.path.join(tmp, "swt_swa.bin")),
           load_f64(os.path.join(tmp, "m_swt_swa.bin")), 1e-11, results)
    record("matlab:swt_swd", load_f64(os.path.join(tmp, "swt_swd.bin")),
           load_f64(os.path.join(tmp, "m_swt_swd.bin")), 1e-11, results)
    record("matlab:modwt_db2", load_f64(os.path.join(tmp, "modwt_db2.bin")),
           load_f64(os.path.join(tmp, "m_modwt_db2.bin")), 1e-11, results)

    # Denoise / threshold / entropy
    record("matlab:wthresh_soft", load_f64(os.path.join(tmp, "thresh_soft.bin")),
           load_f64(os.path.join(tmp, "m_thresh_soft.bin")), 1e-15, results)
    record("matlab:wthresh_hard", load_f64(os.path.join(tmp, "thresh_hard.bin")),
           load_f64(os.path.join(tmp, "m_thresh_hard.bin")), 1e-15, results)
    record("matlab:wnoisest", load_f64(os.path.join(tmp, "noise_mad.bin")),
           load_f64(os.path.join(tmp, "m_noise_mad.bin")), 1e-14, results)
    record("matlab:wentropy_all5", load_f64(os.path.join(tmp, "entropies.bin")),
           load_f64(os.path.join(tmp, "m_entropies.bin")), 1e-13, results)

    # All 9 wextend modes (lf > n)
    for m in EXT_MODES:
        record(f"matlab:wextend_{m}", load_f64(os.path.join(tmp, f"wext_{m}.bin")),
               load_f64(os.path.join(tmp, f"m_wext_{m}.bin")), 1e-13, results)

    # Continuous shapes, centfrq, wavefun_psi
    record("matlab:mexihat", load_f64(os.path.join(tmp, "shape_mexihat.bin")),
           load_f64(os.path.join(tmp, "m_shape_mexihat.bin")), 1e-13, results)
    record("matlab:morlet", load_f64(os.path.join(tmp, "shape_morlet.bin")),
           load_f64(os.path.join(tmp, "m_shape_morlet.bin")), 1e-13, results)
    record("matlab:meyeraux", load_f64(os.path.join(tmp, "shape_meyeraux.bin")),
           load_f64(os.path.join(tmp, "m_shape_meyeraux.bin")), 1e-14, results)
    record("matlab:shanwavf", load_c64(os.path.join(tmp, "shape_shan.bin")),
           load_c64(os.path.join(tmp, "m_shape_shan.bin")), 1e-12, results)
    record("matlab:cmorwavf", load_c64(os.path.join(tmp, "shape_cmor.bin")),
           load_c64(os.path.join(tmp, "m_shape_cmor.bin")), 1e-13, results)
    record("matlab:fbspwavf", load_c64(os.path.join(tmp, "shape_fbsp.bin")),
           load_c64(os.path.join(tmp, "m_shape_fbsp.bin")), 1e-12, results)
    for ord_p in range(1, 9):
        record(f"matlab:gauswavf_{ord_p}", load_f64(os.path.join(tmp, f"shape_gaus{ord_p}.bin")),
               load_f64(os.path.join(tmp, f"m_shape_gaus{ord_p}.bin")), 1e-12, results)
        record(f"matlab:cgauwavf_{ord_p}", load_c64(os.path.join(tmp, f"shape_cgau{ord_p}.bin")),
               load_c64(os.path.join(tmp, f"m_shape_cgau{ord_p}.bin")), 1e-12, results)

    record("matlab:centfrq_9waves", load_f64(os.path.join(tmp, "centfrqs.bin")),
           load_f64(os.path.join(tmp, "m_centfrqs.bin")), 1e-12, results)
    record("matlab:wavefun_db4_psi", load_f64(os.path.join(tmp, "wavefun_db4_psi.bin")),
           load_f64(os.path.join(tmp, "m_wavefun_db4_psi.bin")), 1e-11, results)

    # CWT (Morse, Amor, Bump): frequencies, COI, and complex coefficients
    for w in ("morse", "amor", "bump"):
        record(f"matlab:cwt_{w}_freqs", load_f64(os.path.join(tmp, f"cwt_{w}_freqs.bin")),
               load_f64(os.path.join(tmp, f"m_cwt_{w}_freqs.bin")), 1e-11, results)
        record(f"matlab:cwt_{w}_coi", load_f64(os.path.join(tmp, f"cwt_{w}_coi.bin")),
               load_f64(os.path.join(tmp, f"m_cwt_{w}_coi.bin")), 1e-10, results)
        record(f"matlab:cwt_{w}_cfs", load_c64(os.path.join(tmp, f"cwt_{w}_cfs.bin")),
               load_c64(os.path.join(tmp, f"m_cwt_{w}_cfs.bin")), 1e-10, results)


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path/to/test_wavelet.exe>", file=sys.stderr)
        sys.exit(2)
    exe = sys.argv[1]
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([exe, "--dump-ref", tmp], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"test_wavelet --dump-ref failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

        verify_mathematical_identities(tmp, results)
        verify_pywt(tmp, results)
        verify_matlab(tmp, results)

    max_rel = max(r for _, r in results) if results else 0.0
    for name, rel in results:
        print(f"  PASS  {name:36s} rel={rel:.2e}")
    print(f"\nALL {len(results)}/{len(results)} REFERENCE CHECKS PASSED (max rel={max_rel:.2e})")


if __name__ == "__main__":
    main()
