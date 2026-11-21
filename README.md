# keel::wavelet

Zero-STL C++20 discrete and stationary wavelet transform library built on [`keel`](https://github.com/keelcpp/keel).

## Features

- **55 built-in filter banks**: `haar`, `db1..db10`, `sym2..sym10`, `coif1..coif5`, `bior1.1..bior6.8`, `rbio1.1..rbio6.8`, plus `orthfilt` and `qmf`
- **1D & 2D DWT**: `dwt`, `idwt`, `dwt2`, `idwt2` (symmetric boundary extension)
- **Multilevel 1D & 2D DWT**: `wavedec`, `waverec`, `appcoef`, `detcoef`, `wrcoef`, `upcoef`, `wenergy`, `wmaxlev`, `wavedec2`, `waverec2`, `appcoef2`, `detcoef2`
- **Haar & dyadic primitives**: `haar_step`/`ihaar_step` (orthogonal & integer lifting, real & complex), `dyaddown_1d`/`dyadup_1d`
- **Undecimated transforms**: `swt_ad` (à trous stationary wavelet transform), `modwt`, `imodwt` (maximal-overlap DWT)
- **Continuous wavelets & CWT**: `mexihat`, `morlet`, `meyeraux`, `shanwavf`, `cmorwavf`, `fbspwavf`, `gauswavf`, `cgauwavf`, `wavefun_psi`, `centfrq`, and analytic 1D `cwt` (Morse, Analytic Morlet, Bump)
- **Denoising & entropy**: `threshold_soft`, `threshold_hard`, `noise_sigma_mad`, `denoise_visushrink`, `entropy` (Shannon, LogEnergy, Threshold, SURE, Norm)
- Zero STL headers in public headers and implementation; full `-fno-exceptions -fno-rtti` support

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
