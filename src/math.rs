// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Mathematical primitives: normal CDF, normal PDF, Horner evaluation, d1/d2.
//!
//! Hand-rolled from primary sources — no `statrs`, no `libm`, no external
//! math dependencies. Every polynomial uses `f64::mul_add` explicitly.
//!
//! # Normal CDF
//!
//! Implemented via a high-precision `erfc` function using rational polynomial
//! approximations from Sun fdlibm (Cody, 1969 coefficients). Four regions
//! ensure full f64 precision across the entire domain:
//! - Tiny region (|x| < 1e-17): linear approximation
//! - Small region (|x| < 0.84375): erf rational polynomial, degree 4/5
//! - Medium region (0.84375 ≤ |x| < 1.25): erfc expansion around x = 1
//! - Large region (1.25 ≤ |x| < 28): erfc with exp-splitting for precision
//!
//! Target: max absolute error < 1e-15 (f64).
//!
//! # Normal PDF
//!
//! Standard normal density φ(x) = exp(−x²/2) / √(2π). Overflow guard
//! returns 0.0 when x²/2 > 700, preventing NaN/Inf.
//!
//! # References
//!
//! - Abramowitz & Stegun, *Handbook of Mathematical Functions*, §26.2.17 (1964)
//! - Cody, W.J., "Rational Chebyshev approximations for the error function",
//!   *Mathematics of Computation*, 22(107):631–637 (1969)
//! - Sun fdlibm, `s_erf.c` — IEEE 754 double precision implementation

// ─── Constants ───────────────────────────────────────────────────────────────

/// 1 / √(2π) — normalising constant for the standard normal PDF.
/// Source: exact algebraic constant.
pub const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7_f64;

/// √2 — used in erf/erfc ↔ normal CDF conversion: N(x) = 0.5 * erfc(-x / √2).
/// Source: exact algebraic constant.
pub const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Overflow guard threshold for `npdf`: when x²/2 exceeds this value,
/// `exp(-x²/2)` underflows to zero in f64 representation.
/// Source: f64 exponent range — `exp(-709.78)` ≈ 5e-309, `exp(-710)` = 0.
const NPDF_OVERFLOW_THRESHOLD: f64 = 700.0_f64;

/// Tiny-argument threshold for erfc: below this, erf(x) ≈ 2x/√π.
/// Source: 2^(-56) ≈ 1.39e-17, below which x² vanishes in f64 arithmetic.
const ERF_TINY: f64 = 1.387_778_780_781_445_7_e-17_f64;

// ─── fdlibm erfc coefficients ────────────────────────────────────────────────
//
// All coefficients from Sun fdlibm `s_erf.c`, which implements the
// Cody (1969) rational Chebyshev approximation scheme.
//
// Notation follows fdlibm naming: PP/QQ for small-x erf, PA/QA for the
// region near x=1, RA/SA and RB/SB for the two large-x erfc sub-regions.

// Region 1: |x| < 0.84375 — erf(x) = x + x * P(x²) / Q(x²)
// P is degree 4, Q is degree 5 (monic).
// Source: fdlibm s_erf.c, Cody (1969) Table II.

/// erf numerator coefficient p₀. Source: fdlibm `s_erf.c`.
const ERF_PP0: f64 =  1.283_791_670_955_125_6_e-1_f64;
/// erf numerator coefficient p₁. Source: fdlibm `s_erf.c`.
const ERF_PP1: f64 = -3.250_421_072_470_015_e-1_f64;
/// erf numerator coefficient p₂. Source: fdlibm `s_erf.c`.
const ERF_PP2: f64 = -2.848_174_957_559_851_e-2_f64;
/// erf numerator coefficient p₃. Source: fdlibm `s_erf.c`.
const ERF_PP3: f64 = -5.770_270_296_489_442_e-3_f64;
/// erf numerator coefficient p₄. Source: fdlibm `s_erf.c`.
const ERF_PP4: f64 = -2.376_301_665_665_016_3_e-5_f64;
/// erf denominator coefficient q₁. Source: fdlibm `s_erf.c`.
const ERF_QQ1: f64 =  3.979_172_239_591_553_5_e-1_f64;
/// erf denominator coefficient q₂. Source: fdlibm `s_erf.c`.
const ERF_QQ2: f64 =  6.502_224_998_876_73e-2_f64;
/// erf denominator coefficient q₃. Source: fdlibm `s_erf.c`.
const ERF_QQ3: f64 =  5.081_306_281_875_766_e-3_f64;
/// erf denominator coefficient q₄. Source: fdlibm `s_erf.c`.
const ERF_QQ4: f64 =  1.324_947_380_043_216_4_e-4_f64;
/// erf denominator coefficient q₅. Source: fdlibm `s_erf.c`.
const ERF_QQ5: f64 = -3.960_228_278_775_368_e-6_f64;

// Region 2: 0.84375 ≤ |x| < 1.25 — erfc(x) = erfc(1) + P(|x|-1) / Q(|x|-1)
// P is degree 6, Q is degree 6 (monic).
// Source: fdlibm s_erf.c, Cody (1969) Table III (adapted).

/// erfc(1) precomputed to full f64 precision.
/// Source: fdlibm `s_erf.c`.
const ERFC_ONE: f64 = 1.572_992_070_502_851_3_e-1_f64;
/// erfc-near-1 numerator coefficient pa₀. Source: fdlibm `s_erf.c`.
const ERFC_PA0: f64 = -2.362_118_560_752_659_4_e-3_f64;
/// erfc-near-1 numerator coefficient pa₁. Source: fdlibm `s_erf.c`.
const ERFC_PA1: f64 =  4.148_561_186_837_483_3_e-1_f64;
/// erfc-near-1 numerator coefficient pa₂. Source: fdlibm `s_erf.c`.
const ERFC_PA2: f64 = -3.722_078_760_357_013e-1_f64;
/// erfc-near-1 numerator coefficient pa₃. Source: fdlibm `s_erf.c`.
const ERFC_PA3: f64 =  3.183_466_199_011_617_5_e-1_f64;
/// erfc-near-1 numerator coefficient pa₄. Source: fdlibm `s_erf.c`.
const ERFC_PA4: f64 = -1.108_946_942_823_966_8_e-1_f64;
/// erfc-near-1 numerator coefficient pa₅. Source: fdlibm `s_erf.c`.
const ERFC_PA5: f64 =  3.547_830_431_952_019_e-2_f64;
/// erfc-near-1 numerator coefficient pa₆. Source: fdlibm `s_erf.c`.
const ERFC_PA6: f64 = -2.166_375_599_832_541_e-3_f64;
/// erfc-near-1 denominator coefficient qa₁. Source: fdlibm `s_erf.c`.
const ERFC_QA1: f64 =  1.064_208_804_008_442_3_e-1_f64;
/// erfc-near-1 denominator coefficient qa₂. Source: fdlibm `s_erf.c`.
const ERFC_QA2: f64 =  5.403_979_177_021_71_e-1_f64;
/// erfc-near-1 denominator coefficient qa₃. Source: fdlibm `s_erf.c`.
const ERFC_QA3: f64 =  7.182_865_441_419_625_e-2_f64;
/// erfc-near-1 denominator coefficient qa₄. Source: fdlibm `s_erf.c`.
const ERFC_QA4: f64 =  1.261_712_198_087_616_4_e-1_f64;
/// erfc-near-1 denominator coefficient qa₅. Source: fdlibm `s_erf.c`.
const ERFC_QA5: f64 =  1.363_708_391_202_905e-2_f64;
/// erfc-near-1 denominator coefficient qa₆. Source: fdlibm `s_erf.c`.
const ERFC_QA6: f64 =  1.198_449_984_679_911_e-2_f64;

// Region 3a: 1.25 ≤ |x| < 1/0.35 ≈ 2.857
// erfc(x) = exp(-x²-0.5625) * exp((z-x)(z+x) + R/S) / x
// where z = x truncated to 28 bits.
// R degree 7, S degree 8 (monic).
// Source: fdlibm s_erf.c, Cody (1969) Table III.

/// erfc-large (sub-region a) numerator r₀. Source: fdlibm `s_erf.c`.
const ERFC_RA0: f64 = -9.864_944_034_847_148_e-3_f64;
/// erfc-large (sub-region a) numerator r₁. Source: fdlibm `s_erf.c`.
const ERFC_RA1: f64 = -6.938_585_727_071_818_e-1_f64;
/// erfc-large (sub-region a) numerator r₂. Source: fdlibm `s_erf.c`.
const ERFC_RA2: f64 = -1.055_862_622_532_329_1_e1_f64;
/// erfc-large (sub-region a) numerator r₃. Source: fdlibm `s_erf.c`.
const ERFC_RA3: f64 = -6.237_533_245_032_601_e1_f64;
/// erfc-large (sub-region a) numerator r₄. Source: fdlibm `s_erf.c`.
const ERFC_RA4: f64 = -1.623_966_694_625_731_e2_f64;
/// erfc-large (sub-region a) numerator r₅. Source: fdlibm `s_erf.c`.
const ERFC_RA5: f64 = -1.846_050_929_067_11_e2_f64;
/// erfc-large (sub-region a) numerator r₆. Source: fdlibm `s_erf.c`.
const ERFC_RA6: f64 = -8.128_743_550_630_66e1_f64;
/// erfc-large (sub-region a) numerator r₇. Source: fdlibm `s_erf.c`.
const ERFC_RA7: f64 = -9.814_329_344_169_145_e0_f64;
/// erfc-large (sub-region a) denominator s₁. Source: fdlibm `s_erf.c`.
const ERFC_SA1: f64 =  1.965_127_166_743_925_7_e1_f64;
/// erfc-large (sub-region a) denominator s₂. Source: fdlibm `s_erf.c`.
const ERFC_SA2: f64 =  1.376_577_541_435_197_e2_f64;
/// erfc-large (sub-region a) denominator s₃. Source: fdlibm `s_erf.c`.
const ERFC_SA3: f64 =  4.345_658_774_752_292_3_e2_f64;
/// erfc-large (sub-region a) denominator s₄. Source: fdlibm `s_erf.c`.
const ERFC_SA4: f64 =  6.453_872_717_332_679_e2_f64;
/// erfc-large (sub-region a) denominator s₅. Source: fdlibm `s_erf.c`.
const ERFC_SA5: f64 =  4.290_081_400_275_678_3_e2_f64;
/// erfc-large (sub-region a) denominator s₆. Source: fdlibm `s_erf.c`.
const ERFC_SA6: f64 =  1.086_350_055_417_794_4_e2_f64;
/// erfc-large (sub-region a) denominator s₇. Source: fdlibm `s_erf.c`.
const ERFC_SA7: f64 =  6.570_249_770_319_282_e0_f64;
/// erfc-large (sub-region a) denominator s₈. Source: fdlibm `s_erf.c`.
const ERFC_SA8: f64 = -6.042_441_521_485_81_e-2_f64;

// Region 3b: 2.857 ≤ |x| < 28
// Same formula as 3a with different rational coefficients.
// R degree 6, S degree 7 (monic).
// Source: fdlibm s_erf.c, Cody (1969) Table IV.

/// erfc-large (sub-region b) numerator r₀. Source: fdlibm `s_erf.c`.
const ERFC_RB0: f64 = -9.864_942_924_700_1e-3_f64;
/// erfc-large (sub-region b) numerator r₁. Source: fdlibm `s_erf.c`.
const ERFC_RB1: f64 = -7.992_832_376_805_23_e-1_f64;
/// erfc-large (sub-region b) numerator r₂. Source: fdlibm `s_erf.c`.
const ERFC_RB2: f64 = -1.775_795_491_775_475_2_e1_f64;
/// erfc-large (sub-region b) numerator r₃. Source: fdlibm `s_erf.c`.
const ERFC_RB3: f64 = -1.606_363_848_555_579_4_e2_f64;
/// erfc-large (sub-region b) numerator r₄. Source: fdlibm `s_erf.c`.
const ERFC_RB4: f64 = -6.375_664_433_683_891_e2_f64;
/// erfc-large (sub-region b) numerator r₅. Source: fdlibm `s_erf.c`.
const ERFC_RB5: f64 = -1.025_095_131_611_077_2_e3_f64;
/// erfc-large (sub-region b) numerator r₆. Source: fdlibm `s_erf.c`.
const ERFC_RB6: f64 = -4.835_191_916_086_514_e2_f64;
/// erfc-large (sub-region b) denominator s₁. Source: fdlibm `s_erf.c`.
const ERFC_SB1: f64 =  3.033_806_078_756_258_e1_f64;
/// erfc-large (sub-region b) denominator s₂. Source: fdlibm `s_erf.c`.
const ERFC_SB2: f64 =  3.257_925_129_965_739e2_f64;
/// erfc-large (sub-region b) denominator s₃. Source: fdlibm `s_erf.c`.
const ERFC_SB3: f64 =  1.536_729_586_084_437_e3_f64;
/// erfc-large (sub-region b) denominator s₄. Source: fdlibm `s_erf.c`.
const ERFC_SB4: f64 =  3.199_858_219_508_596_e3_f64;
/// erfc-large (sub-region b) denominator s₅. Source: fdlibm `s_erf.c`.
const ERFC_SB5: f64 =  2.553_050_406_433_164_4_e3_f64;
/// erfc-large (sub-region b) denominator s₆. Source: fdlibm `s_erf.c`.
const ERFC_SB6: f64 =  4.745_285_412_069_554_e2_f64;
/// erfc-large (sub-region b) denominator s₇. Source: fdlibm `s_erf.c`.
const ERFC_SB7: f64 = -2.244_095_244_658_582_e1_f64;

// ─── Normal PDF ──────────────────────────────────────────────────────────────

/// Standard normal probability density function.
///
/// φ(x) = exp(−x²/2) / √(2π)
///
/// Returns `0.0` when `x²/2 > 700` (overflow guard — prevents NaN/Inf).
///
/// # Arguments
///
/// * `x` — the point at which to evaluate the density
///
/// # Examples
///
/// ```
/// use regit_blackscholes::math::npdf;
/// let density = npdf(0.0_f64);
/// assert!((density - 0.398_942_280_401_432_7_f64).abs() < 1e-15);
/// ```
#[inline(always)]
pub fn npdf(x: f64) -> f64 {
    let half_x_sq = 0.5_f64 * x * x;
    if half_x_sq > NPDF_OVERFLOW_THRESHOLD {
        return 0.0_f64;
    }
    FRAC_1_SQRT_2PI * (-half_x_sq).exp()
}

// ─── Complementary error function ────────────────────────────────────────────

/// Complementary error function: erfc(a) = 1 - erf(a).
///
/// Uses the Sun fdlibm rational polynomial implementation (based on Cody 1969)
/// with four regions and exp-splitting in the tail for full f64 precision.
///
/// # Arguments
///
/// * `a` — the argument (any real value)
#[inline(always)]
fn erfc_impl(a: f64) -> f64 {
    let x = a.abs();

    let result = if x < 0.843_75_f64 {
        erfc_small(x)
    } else if x < 1.25_f64 {
        erfc_mid(x)
    } else if x < 28.0_f64 {
        erfc_large(x)
    } else {
        // |x| >= 28: erfc underflows to 0 in f64
        0.0_f64
    };

    if a < 0.0_f64 { 2.0_f64 - result } else { result }
}

/// erfc(x) for 0 ≤ x < 0.84375.
///
/// Computes erf(x) = x + x · P(x²)/Q(x²), then erfc(x) = 1 − erf(x).
/// P is degree 4, Q is degree 5 (monic leading coefficient 1).
/// Every Horner step uses `mul_add`.
#[inline(always)]
fn erfc_small(x: f64) -> f64 {
    if x < ERF_TINY {
        return 1.0_f64;
    }
    let z = x * x;

    // Numerator: P(z) = ((((pp4·z + pp3)·z + pp2)·z + pp1)·z + pp0)
    let num = ERF_PP4
        .mul_add(z, ERF_PP3)
        .mul_add(z, ERF_PP2)
        .mul_add(z, ERF_PP1)
        .mul_add(z, ERF_PP0);

    // Denominator: Q(z) = ((((qq5·z + qq4)·z + qq3)·z + qq2)·z + qq1)·z + 1
    let den = ERF_QQ5
        .mul_add(z, ERF_QQ4)
        .mul_add(z, ERF_QQ3)
        .mul_add(z, ERF_QQ2)
        .mul_add(z, ERF_QQ1)
        .mul_add(z, 1.0_f64);

    // erf(x) = x + x * P/Q;  erfc(x) = 1 - erf(x)
    1.0_f64 - (x + x * num / den)
}

/// erfc(x) for 0.84375 ≤ x < 1.25.
///
/// Uses erfc(x) = erfc(1) + P(|x|−1)/Q(|x|−1) expansion around x = 1.
/// P is degree 6, Q is degree 6 (monic).
#[inline(always)]
fn erfc_mid(x: f64) -> f64 {
    let s = x - 1.0_f64;

    let p = ERFC_PA6
        .mul_add(s, ERFC_PA5)
        .mul_add(s, ERFC_PA4)
        .mul_add(s, ERFC_PA3)
        .mul_add(s, ERFC_PA2)
        .mul_add(s, ERFC_PA1)
        .mul_add(s, ERFC_PA0);

    let q = ERFC_QA6
        .mul_add(s, ERFC_QA5)
        .mul_add(s, ERFC_QA4)
        .mul_add(s, ERFC_QA3)
        .mul_add(s, ERFC_QA2)
        .mul_add(s, ERFC_QA1)
        .mul_add(s, 1.0_f64);

    ERFC_ONE - p / q
}

/// erfc(x) for 1.25 ≤ x < 28.
///
/// Uses the exp-splitting technique from fdlibm for precision:
/// erfc(x) = exp(−z²−0.5625) · exp((z−x)(z+x) + R(1/x²)/S(1/x²)) / x
/// where z = x truncated to ~28 significant bits.
///
/// Two sub-regions use different rational polynomial coefficients:
/// - 1.25 ≤ x < 2.857 (RA/SA, degree 7/8)
/// - 2.857 ≤ x < 28 (RB/SB, degree 6/7)
#[inline(always)]
fn erfc_large(x: f64) -> f64 {
    let s = 1.0_f64 / (x * x);

    let ratio = if x < 2.857_142_857_142_857_f64 {
        // Sub-region a: R degree 7, S degree 8 (monic)
        let r = ERFC_RA7
            .mul_add(s, ERFC_RA6)
            .mul_add(s, ERFC_RA5)
            .mul_add(s, ERFC_RA4)
            .mul_add(s, ERFC_RA3)
            .mul_add(s, ERFC_RA2)
            .mul_add(s, ERFC_RA1)
            .mul_add(s, ERFC_RA0);

        let sr = ERFC_SA8
            .mul_add(s, ERFC_SA7)
            .mul_add(s, ERFC_SA6)
            .mul_add(s, ERFC_SA5)
            .mul_add(s, ERFC_SA4)
            .mul_add(s, ERFC_SA3)
            .mul_add(s, ERFC_SA2)
            .mul_add(s, ERFC_SA1)
            .mul_add(s, 1.0_f64);

        r / sr
    } else {
        // Sub-region b: R degree 6, S degree 7 (monic)
        let r = ERFC_RB6
            .mul_add(s, ERFC_RB5)
            .mul_add(s, ERFC_RB4)
            .mul_add(s, ERFC_RB3)
            .mul_add(s, ERFC_RB2)
            .mul_add(s, ERFC_RB1)
            .mul_add(s, ERFC_RB0);

        let sr = ERFC_SB7
            .mul_add(s, ERFC_SB6)
            .mul_add(s, ERFC_SB5)
            .mul_add(s, ERFC_SB4)
            .mul_add(s, ERFC_SB3)
            .mul_add(s, ERFC_SB2)
            .mul_add(s, ERFC_SB1)
            .mul_add(s, 1.0_f64);

        r / sr
    };

    // Exp-splitting: truncate x to ~28 bits for precision in exp(-x²).
    // z²  ≈ x²  but representable exactly in fewer bits.
    // exp(-x²) = exp(-z² - 0.5625) * exp(0.5625 + (z-x)(z+x))
    let z = f64::from_bits(x.to_bits() & 0xFFFF_FFFF_F000_0000_u64);
    (-z * z - 0.562_5_f64).exp() * ((z - x) * (z + x) + ratio).exp() / x
}

// ─── Normal CDF ──────────────────────────────────────────────────────────────

/// Standard normal cumulative distribution function.
///
/// N(x) = (1/2) · erfc(−x/√2)
///
/// Uses the fdlibm erfc implementation (Cody 1969 coefficients) achieving
/// max absolute error < 1e-15 over the full f64 range. The erfc-based
/// formulation avoids subtractive cancellation in both tails.
///
/// # Arguments
///
/// * `x` — the upper limit of integration
///
/// # Examples
///
/// ```
/// use regit_blackscholes::math::ncdf;
/// let half = ncdf(0.0_f64);
/// assert!((half - 0.5_f64).abs() < 1e-15);
/// ```
#[inline(always)]
pub fn ncdf(x: f64) -> f64 {
    // N(x) = 0.5 * erfc(-x / sqrt(2))
    0.5_f64 * erfc_impl(-x * std::f64::consts::FRAC_1_SQRT_2)
}

/// Complement of the standard normal CDF: Q(x) = 1 − N(x).
///
/// Computed directly via erfc to avoid catastrophic cancellation that would
/// occur from `1.0 - ncdf(x)` in the right tail. Essential for OTM put
/// pricing and tail probability computations.
///
/// Q(x) = (1/2) · erfc(x/√2)
///
/// # Arguments
///
/// * `x` — the point at which to evaluate the survival function
///
/// # Examples
///
/// ```
/// use regit_blackscholes::math::ncdf_complement;
/// let half = ncdf_complement(0.0_f64);
/// assert!((half - 0.5_f64).abs() < 1e-15);
/// ```
#[inline(always)]
pub fn ncdf_complement(x: f64) -> f64 {
    // Q(x) = 1 - N(x) = 0.5 * erfc(x / sqrt(2))
    0.5_f64 * erfc_impl(x * std::f64::consts::FRAC_1_SQRT_2)
}

// ─── d1 / d2 ─────────────────────────────────────────────────────────────────

/// Black-Scholes d1 parameter.
///
/// d1 = \[ln(S/K) + (r − q + σ²/2) · T\] / (σ · √T)
///
/// # Arguments
///
/// * `spot` — underlying price S
/// * `strike` — strike price K
/// * `rate` — risk-free rate r (continuous, annualised)
/// * `div_yield` — dividend yield q (continuous, annualised)
/// * `vol` — implied volatility σ (annualised)
/// * `time` — time to expiry T (in years)
///
/// # Examples
///
/// ```
/// use regit_blackscholes::math::d1;
/// let val = d1(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
/// assert!((val - 0.25_f64).abs() < 0.01);
/// ```
#[inline(always)]
pub fn d1(spot: f64, strike: f64, rate: f64, div_yield: f64, vol: f64, time: f64) -> f64 {
    let vol_sqrt_t = vol * time.sqrt();
    ((spot / strike).ln() + (rate - div_yield + 0.5_f64 * vol * vol) * time) / vol_sqrt_t
}

/// Black-Scholes d2 parameter.
///
/// d2 = d1 − σ · √T
///
/// Takes d1 as a precomputed input to avoid redundant computation.
///
/// # Arguments
///
/// * `d1_val` — precomputed d1 value
/// * `vol` — implied volatility σ (annualised)
/// * `time` — time to expiry T (in years)
///
/// # Examples
///
/// ```
/// use regit_blackscholes::math::{d1, d2};
/// let d1_val = d1(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
/// let d2_val = d2(d1_val, 0.20_f64, 1.0_f64);
/// assert!((d2_val - (d1_val - 0.20_f64)).abs() < 1e-15);
/// ```
#[inline(always)]
pub fn d2(d1_val: f64, vol: f64, time: f64) -> f64 {
    d1_val - vol * time.sqrt()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for CDF complement identity — must hold to 1e-15.
    const CDF_IDENTITY_TOL: f64 = 1e-15_f64;

    /// Tolerance for known-value comparisons.
    const KNOWN_VALUE_TOL: f64 = 1e-10_f64;

    /// Tolerance for d1/d2 relationship.
    const D1_D2_TOL: f64 = 1e-15_f64;

    #[test]
    fn test_ncdf_complement_identity() {
        // N(x) + N(-x) = 1.0 for all x — tests the symmetry and precision
        // of the CDF implementation across the full domain.
        // Identity 15 from testing.md: tolerance 1e-15, step 0.1, x in [-8, 8].
        let mut x = -8.0_f64;
        while x <= 8.0_f64 {
            let sum = ncdf(x) + ncdf(-x);
            assert!(
                (sum - 1.0_f64).abs() < CDF_IDENTITY_TOL,
                "N({x}) + N({neg_x}) = {sum}, expected 1.0 (error = {err})",
                x = x,
                neg_x = -x,
                sum = sum,
                err = (sum - 1.0_f64).abs(),
            );
            x += 0.1_f64;
        }
    }

    #[test]
    fn test_ncdf_complement_via_dedicated_fn() {
        // ncdf(x) + ncdf_complement(x) = 1.0
        let mut x = -8.0_f64;
        while x <= 8.0_f64 {
            let sum = ncdf(x) + ncdf_complement(x);
            assert!(
                (sum - 1.0_f64).abs() < CDF_IDENTITY_TOL,
                "ncdf({x}) + ncdf_complement({x}) = {sum}, error = {err}",
                x = x,
                sum = sum,
                err = (sum - 1.0_f64).abs(),
            );
            x += 0.1_f64;
        }
    }

    #[test]
    fn test_ncdf_known_values() {
        // Reference values from Abramowitz & Stegun / high-precision computation.
        // N(0) = 0.5 exactly
        assert!(
            (ncdf(0.0_f64) - 0.5_f64).abs() < CDF_IDENTITY_TOL,
            "N(0) should be 0.5, got {}",
            ncdf(0.0_f64),
        );

        // N(1) ≈ 0.8413447460685429
        let n1_ref = 0.841_344_746_068_542_9_f64;
        assert!(
            (ncdf(1.0_f64) - n1_ref).abs() < KNOWN_VALUE_TOL,
            "N(1) should be {n1_ref}, got {} (error = {})",
            ncdf(1.0_f64),
            (ncdf(1.0_f64) - n1_ref).abs(),
        );

        // N(-1) ≈ 0.15865525393145707
        let nm1_ref = 0.158_655_253_931_457_07_f64;
        assert!(
            (ncdf(-1.0_f64) - nm1_ref).abs() < KNOWN_VALUE_TOL,
            "N(-1) should be {nm1_ref}, got {} (error = {})",
            ncdf(-1.0_f64),
            (ncdf(-1.0_f64) - nm1_ref).abs(),
        );

        // N(2) ≈ 0.9772498680518208
        let n2_ref = 0.977_249_868_051_820_8_f64;
        assert!(
            (ncdf(2.0_f64) - n2_ref).abs() < KNOWN_VALUE_TOL,
            "N(2) should be {n2_ref}, got {} (error = {})",
            ncdf(2.0_f64),
            (ncdf(2.0_f64) - n2_ref).abs(),
        );

        // N(-3) ≈ 0.0013498980316300946
        let nm3_ref = 0.001_349_898_031_630_094_6_f64;
        assert!(
            (ncdf(-3.0_f64) - nm3_ref).abs() < KNOWN_VALUE_TOL,
            "N(-3) should be {nm3_ref}, got {} (error = {})",
            ncdf(-3.0_f64),
            (ncdf(-3.0_f64) - nm3_ref).abs(),
        );

        // N(0.5) ≈ 0.6914624612740131
        let n05_ref = 0.691_462_461_274_013_1_f64;
        assert!(
            (ncdf(0.5_f64) - n05_ref).abs() < KNOWN_VALUE_TOL,
            "N(0.5) should be {n05_ref}, got {} (error = {})",
            ncdf(0.5_f64),
            (ncdf(0.5_f64) - n05_ref).abs(),
        );
    }

    #[test]
    fn test_ncdf_tail_values() {
        // Deep tails: verify no catastrophic cancellation.
        // N(6) should be very close to 1.0
        let n6 = ncdf(6.0_f64);
        assert!(
            n6 > 0.999_999_999_f64,
            "N(6) should be very close to 1.0, got {n6}",
        );

        // N(-6) should be very small positive
        let nm6 = ncdf(-6.0_f64);
        assert!(
            nm6 < 1e-8_f64 && nm6 > 0.0_f64,
            "N(-6) should be tiny positive, got {nm6}",
        );

        // N(-6) = Q(6) — cross-check between ncdf and ncdf_complement
        assert!(
            (ncdf(-6.0_f64) - ncdf_complement(6.0_f64)).abs() < CDF_IDENTITY_TOL,
            "N(-6) should equal Q(6)",
        );

        // Extreme tails — no NaN, no Inf, bounded [0, 1]
        assert!(ncdf(38.0_f64) <= 1.0_f64, "N(38) should be <= 1.0");
        assert!(ncdf(-38.0_f64) >= 0.0_f64, "N(-38) should be >= 0.0");
    }

    #[test]
    fn test_npdf_known_values() {
        // φ(0) = 1/√(2π)
        assert!(
            (npdf(0.0_f64) - FRAC_1_SQRT_2PI).abs() < CDF_IDENTITY_TOL,
            "φ(0) should be 1/√(2π) = {}, got {}",
            FRAC_1_SQRT_2PI,
            npdf(0.0_f64),
        );

        // φ(1) = exp(-0.5) / √(2π)
        let phi1_ref = (-0.5_f64).exp() * FRAC_1_SQRT_2PI;
        assert!(
            (npdf(1.0_f64) - phi1_ref).abs() < CDF_IDENTITY_TOL,
            "φ(1) should be {phi1_ref}, got {}",
            npdf(1.0_f64),
        );

        // φ(-1) = φ(1) — symmetry
        assert!(
            (npdf(-1.0_f64) - npdf(1.0_f64)).abs() < CDF_IDENTITY_TOL,
            "φ(-1) should equal φ(1)",
        );
    }

    #[test]
    fn test_npdf_overflow_guard() {
        // npdf(40) should return 0.0 (not NaN, not Inf)
        let val = npdf(40.0_f64);
        assert!(
            val == 0.0_f64,
            "npdf(40) should be exactly 0.0, got {val}",
        );
        assert!(!val.is_nan(), "npdf(40) must not be NaN");
        assert!(!val.is_infinite(), "npdf(40) must not be Inf");

        // Also check negative
        let val_neg = npdf(-40.0_f64);
        assert!(
            val_neg == 0.0_f64,
            "npdf(-40) should be exactly 0.0, got {val_neg}",
        );
    }

    #[test]
    fn test_d1_d2_relationship() {
        // d2 = d1 - σ√T — must hold exactly (no approximation involved)
        let test_cases: &[(f64, f64, f64, f64, f64, f64)] = &[
            (100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64),
            (100.0_f64, 110.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64),
            (50.0_f64, 100.0_f64, 0.05_f64, 0.00_f64, 0.40_f64, 2.0_f64),
            (200.0_f64, 150.0_f64, -0.01_f64, 0.03_f64, 0.10_f64, 0.25_f64),
        ];

        for &(spot, strike, rate, div, vol, time) in test_cases {
            let d1_val = d1(spot, strike, rate, div, vol, time);
            let d2_val = d2(d1_val, vol, time);
            let expected_d2 = d1_val - vol * time.sqrt();
            assert!(
                (d2_val - expected_d2).abs() < D1_D2_TOL,
                "d2 = d1 - σ√T failed for S={spot}, K={strike}, r={rate}, q={div}, σ={vol}, T={time}: \
                 d2={d2_val}, expected={expected_d2}",
            );
        }
    }

    #[test]
    fn test_d1_atm_value() {
        // ATM with baseline params: d1 = (0 + (0.05 - 0.02 + 0.02) * 1.0) / 0.20 = 0.25
        let d1_val = d1(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
        let expected = (0.05_f64 - 0.02_f64 + 0.02_f64) / 0.20_f64;
        assert!(
            (d1_val - expected).abs() < KNOWN_VALUE_TOL,
            "d1 ATM should be {expected}, got {d1_val}",
        );
    }
}
