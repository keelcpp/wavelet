# keel::wavelet

Zero-STL C++20 discrete, stationary, and continuous wavelet transform library built on [`keel`](https://github.com/keelcpp/keel) and [`keel::fft`](https://github.com/keelcpp/fft).

## Features

- **55 built-in filter banks**: `haar`, `db1..db10`, `sym2..sym10`, `coif1..coif5`, `bior1.1..bior6.8`, `rbio1.1..rbio6.8`, plus `orthfilt` and in-place-safe `qmf`
- **1D & 2D DWT**: `dwt`, `idwt`, `dwt2`, `idwt2` with periodic half-point symmetric boundary extension (handles arbitrary signal lengths including `n < lf` and single-branch `idwt`)
- **Multilevel 1D & 2D DWT**: `wavedec`, `waverec`, `appcoef`, `detcoef`, `wrcoef`, `upcoef`, overflow-safe `wenergy`, exact integer `wmaxlev`, `wavedec2`, `waverec2`, `appcoef2`, `detcoef2`
- **Haar & dyadic primitives**: `haar_step`/`ihaar_step` (orthogonal & integer lifting, real & complex), `dyaddown_1d`/in-place-safe `dyadup_1d`
- **Undecimated transforms**: O(`N * lf`) à trous `swt_ad`, overflow-free `modwt` / `imodwt`
- **Continuous wavelets & CWT**: `mexihat`, `morlet`, `meyeraux`, `shanwavf`, `cmorwavf`, `fbspwavf`, `gauswavf`, `cgauwavf`, `wavefun_psi`, `centfrq`, and analytic 1D `cwt` (Morse, Analytic Morlet, Bump)
- **Denoising & entropy**: NaN-propagating `threshold_soft`, `threshold_hard`, `noise_sigma_mad`, `denoise_visushrink`, `entropy` (Shannon, LogEnergy, Threshold, SURE, Norm)
- **Boundary extension**: `wextend_1d` / `wextend_1d_c64` supporting all 9 modes (`sym`, `symw`, `asym`, `asymw`, `sp0`, `sp1`, `per`, `zpd`, `ppd`) for arbitrary extension lengths `lf >= 0`
- Zero STL headers in public headers and implementation; full `-fno-exceptions -fno-rtti` support

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python tests/verify_reference.py build/Release/test_wavelet.exe
```
