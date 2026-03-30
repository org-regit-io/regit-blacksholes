// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Core types for option parameterisation and pricing output.
//!
//! All types are generic over `F: Float` to support both `f32` and `f64`
//! paths. Monomorphisation at compile time — no runtime dispatch overhead.

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::errors::{IvError, PricingError};
use crate::iv::IvSolver;
use crate::models::bachelier::BachelierParams;
use crate::models::black76::Black76Params;
use crate::models::displaced::DisplacedParams;

/// Minimal float trait — satisfied by `f32` and `f64`.
///
/// Provides the arithmetic and mathematical operations needed by the
/// pricing engine without pulling in any external dependency. Every
/// bound maps directly to a `std` primitive method.
pub trait Float:
    Copy
    + PartialOrd
    + fmt::Debug
    + fmt::Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The zero value (`0.0`).
    fn zero() -> Self;

    /// The one value (`1.0`).
    fn one() -> Self;

    /// Natural logarithm.
    fn ln(self) -> Self;

    /// Exponential function.
    fn exp(self) -> Self;

    /// Square root.
    fn sqrt(self) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;

    /// Fused multiply-add: `self * a + b`.
    ///
    /// Maps to the hardware FMA instruction when available.
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Converts from an `f64` constant. Used for embedding typed literals.
    fn from_f64(val: f64) -> Self;

    /// Converts to `f64`. Used for error reporting.
    fn to_f64(self) -> f64;

    /// The mathematical constant pi.
    fn pi() -> Self;

    /// Returns `true` if the value is NaN.
    fn is_nan(self) -> bool;

    /// Returns `true` if the value is infinite.
    fn is_infinite(self) -> bool;
}

impl Float for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0_f64
    }

    #[inline(always)]
    fn one() -> Self {
        1.0_f64
    }

    #[inline(always)]
    fn ln(self) -> Self {
        f64::ln(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f64::exp(self)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }

    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        val
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline(always)]
    fn pi() -> Self {
        core::f64::consts::PI
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        f64::is_infinite(self)
    }
}

impl Float for f32 {
    #[inline(always)]
    fn zero() -> Self {
        0.0_f32
    }

    #[inline(always)]
    fn one() -> Self {
        1.0_f32
    }

    #[inline(always)]
    fn ln(self) -> Self {
        f32::ln(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f32::exp(self)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }

    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        val as f32
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[inline(always)]
    fn pi() -> Self {
        core::f32::consts::PI
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        f32::is_infinite(self)
    }
}

/// European option type — call or put.
///
/// Determines the payoff direction in all pricing models.
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::OptionType;
///
/// let ot = OptionType::Call;
/// assert!(matches!(ot, OptionType::Call));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType {
    /// Right to **buy** the underlying at the strike price.
    Call,
    /// Right to **sell** the underlying at the strike price.
    Put,
}

/// Parameters describing a European option contract.
///
/// Generic over `F: Float` to support both `f32` and `f64` paths.
/// All fields use continuous, annualised conventions.
///
/// # Fields
///
/// | Field | Symbol | Description |
/// |-------|--------|-------------|
/// | `spot` | S | Underlying price |
/// | `strike` | K | Strike price |
/// | `rate` | r | Risk-free rate (continuous, annualised) |
/// | `div_yield` | q | Dividend yield (continuous, annualised) |
/// | `vol` | sigma | Implied volatility (annualised) |
/// | `time` | T | Time to expiry in years |
/// | `option_type` | — | Call or Put |
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::{OptionParams, OptionType};
///
/// let params = OptionParams {
///     option_type: OptionType::Call,
///     spot:        100.0_f64,
///     strike:      100.0_f64,
///     rate:        0.05_f64,
///     div_yield:   0.02_f64,
///     vol:         0.20_f64,
///     time:        1.0_f64,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct OptionParams<F: Float> {
    /// Option type — call or put.
    pub option_type: OptionType,
    /// S — underlying spot price.
    pub spot: F,
    /// K — strike price.
    pub strike: F,
    /// r — risk-free rate (continuous, annualised).
    pub rate: F,
    /// q — continuous dividend yield (annualised).
    pub div_yield: F,
    /// sigma — implied volatility (annualised).
    pub vol: F,
    /// T — time to expiry in years.
    pub time: F,
}

/// All 17 analytic Greeks through 3rd order.
///
/// Computed once per pricing call from shared intermediates `d1`, `d2`,
/// `N(d1)`, `N(d2)`, `phi(d1)` — no recomputation.
///
/// # Greek summary
///
/// | Order | Greeks |
/// |-------|--------|
/// | 1st   | Delta, Theta, Vega, Rho, Epsilon, Lambda, Dual Delta |
/// | 2nd   | Gamma, Vanna, Charm, Veta, Vomma, Dual Gamma |
/// | 3rd   | Speed, Zomma, Color, Ultima |
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::Greeks;
///
/// let g = Greeks {
///     delta: 0.5987_f64, gamma: 0.0185_f64, theta: -0.0152_f64,
///     vega: 0.3702_f64, rho: 0.4174_f64, epsilon: -0.5702_f64,
///     lambda: 6.47_f64, vanna: -0.1314_f64, charm: -0.0265_f64,
///     veta: 0.0_f64, vomma: 0.1499_f64, speed: -0.0006_f64,
///     zomma: -0.0009_f64, color: 0.0_f64, ultima: 0.0_f64,
///     dual_delta: -0.4741_f64, dual_gamma: 0.0_f64,
/// };
/// assert!(g.delta > 0.0_f64);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Greeks<F: Float> {
    /// Delta (1st order) — rate of change of price with respect to spot.
    pub delta: F,
    /// Gamma (2nd order) — rate of change of delta with respect to spot.
    pub gamma: F,
    /// Theta (1st order) — rate of change of price with respect to time.
    /// Expressed per calendar day (divide annual theta by 365).
    pub theta: F,
    /// Vega (1st order) — rate of change of price with respect to volatility.
    pub vega: F,
    /// Rho (1st order) — rate of change of price with respect to interest rate.
    pub rho: F,
    /// Epsilon (1st order) — rate of change of price with respect to dividend yield.
    pub epsilon: F,
    /// Lambda (1st order) — percentage change in price per percentage change in spot.
    /// Also known as the elasticity or leverage.
    pub lambda: F,
    /// Vanna (2nd order) — cross-derivative of price with respect to spot and volatility.
    /// Equivalently: `d(Delta)/d(sigma)` or `d(Vega)/d(S)`.
    pub vanna: F,
    /// Charm (2nd order) — rate of change of delta with respect to time.
    /// Also known as delta decay.
    pub charm: F,
    /// Veta (2nd order) — rate of change of vega with respect to time.
    /// Also known as vega decay.
    pub veta: F,
    /// Vomma (2nd order) — rate of change of vega with respect to volatility.
    /// Also known as volga or vega convexity.
    pub vomma: F,
    /// Speed (3rd order) — rate of change of gamma with respect to spot.
    pub speed: F,
    /// Zomma (3rd order) — rate of change of gamma with respect to volatility.
    pub zomma: F,
    /// Color (3rd order) — rate of change of gamma with respect to time.
    /// Also known as gamma decay.
    pub color: F,
    /// Ultima (3rd order) — third derivative of price with respect to volatility.
    pub ultima: F,
    /// Dual delta (1st order) — rate of change of price with respect to strike.
    pub dual_delta: F,
    /// Dual gamma (2nd order) — second derivative of price with respect to strike.
    pub dual_gamma: F,
}

// ─── Ergonomic traits ───────────────────────────────────────────────────────

/// Trait for computing option prices.
///
/// Implemented on [`OptionParams<f64>`], [`Black76Params`], [`BachelierParams`],
/// [`DisplacedParams`], and [`Model`] for a uniform pricing API across all
/// four models.
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::{OptionParams, OptionType, Pricing};
///
/// let params = OptionParams {
///     option_type: OptionType::Call,
///     spot: 100.0_f64, strike: 100.0_f64,
///     rate: 0.05_f64, div_yield: 0.02_f64,
///     vol: 0.20_f64, time: 1.0_f64,
/// };
/// let price = params.price().unwrap();
/// assert!(price > 0.0_f64);
/// ```
pub trait Pricing {
    /// Computes the fair value of the option under the model.
    ///
    /// # Errors
    ///
    /// Returns [`PricingError`] when input validation fails.
    fn price(&self) -> Result<f64, PricingError>;
}

/// Trait for computing all 17 analytic Greeks.
///
/// Currently implemented only for [`OptionParams<f64>`] (Black-Scholes-Merton).
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::{OptionParams, OptionType, GreeksCalc};
///
/// let params = OptionParams {
///     option_type: OptionType::Call,
///     spot: 100.0_f64, strike: 100.0_f64,
///     rate: 0.05_f64, div_yield: 0.02_f64,
///     vol: 0.20_f64, time: 1.0_f64,
/// };
/// let greeks = params.greeks().unwrap();
/// assert!(greeks.delta > 0.0_f64);
/// ```
pub trait GreeksCalc {
    /// Computes all 17 analytic Greeks from shared intermediates.
    ///
    /// # Errors
    ///
    /// Returns [`PricingError`] when input validation fails.
    fn greeks(&self) -> Result<Greeks<f64>, PricingError>;
}

/// Trait for recovering implied volatility from a market price.
///
/// Currently implemented only for [`OptionParams<f64>`] (Black-Scholes-Merton).
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::{OptionParams, OptionType, ImpliedVol};
/// use regit_blackscholes::iv::IvSolver;
///
/// let params = OptionParams {
///     option_type: OptionType::Call,
///     spot: 100.0_f64, strike: 100.0_f64,
///     rate: 0.05_f64, div_yield: 0.02_f64,
///     vol: 0.0_f64, time: 1.0_f64,
/// };
/// let iv = params.implied_vol(9.2270_f64, IvSolver::Auto).unwrap();
/// assert!((iv - 0.20_f64).abs() < 1e-4_f64);
/// ```
pub trait ImpliedVol {
    /// Recovers the implied volatility that reprices the market price.
    ///
    /// # Errors
    ///
    /// Returns [`IvError`] when convergence fails or the market price
    /// is inconsistent with the model.
    fn implied_vol(&self, market_price: f64, solver: IvSolver) -> Result<f64, IvError>;
}

// ─── Trait implementations ──────────────────────────────────────────────────

impl Pricing for OptionParams<f64> {
    #[inline(always)]
    fn price(&self) -> Result<f64, PricingError> {
        crate::models::black_scholes::price(self)
    }
}

impl GreeksCalc for OptionParams<f64> {
    #[inline(always)]
    fn greeks(&self) -> Result<Greeks<f64>, PricingError> {
        crate::greeks::compute_greeks(self)
    }
}

impl ImpliedVol for OptionParams<f64> {
    #[inline(always)]
    fn implied_vol(&self, market_price: f64, solver: IvSolver) -> Result<f64, IvError> {
        crate::iv::implied_vol(self, market_price, solver)
    }
}

impl Pricing for Black76Params {
    #[inline(always)]
    fn price(&self) -> Result<f64, PricingError> {
        crate::models::black76::price(self)
    }
}

impl Pricing for BachelierParams {
    #[inline(always)]
    fn price(&self) -> Result<f64, PricingError> {
        crate::models::bachelier::price(self)
    }
}

impl Pricing for DisplacedParams {
    #[inline(always)]
    fn price(&self) -> Result<f64, PricingError> {
        crate::models::displaced::price(self)
    }
}

// ─── Model enum ─────────────────────────────────────────────────────────────

/// Dynamic dispatch across the four pricing models.
///
/// Wraps [`OptionParams<f64>`], [`Black76Params`], [`BachelierParams`], and
/// [`DisplacedParams`] in a single enum for runtime model selection.
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::{Model, OptionParams, OptionType, Pricing};
/// use regit_blackscholes::models::black76::Black76Params;
///
/// let bs = Model::BlackScholes(OptionParams {
///     option_type: OptionType::Call,
///     spot: 100.0_f64, strike: 100.0_f64,
///     rate: 0.05_f64, div_yield: 0.02_f64,
///     vol: 0.20_f64, time: 1.0_f64,
/// });
/// let price = bs.price().unwrap();
/// assert!(price > 0.0_f64);
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Model {
    /// Black-Scholes-Merton (Merton 1973) — vanilla European, continuous dividend.
    BlackScholes(OptionParams<f64>),
    /// Black-76 (Black 1976) — options on futures/forwards.
    Black76(Black76Params),
    /// Bachelier / Normal model (Bachelier 1900) — for rates near/below zero.
    Bachelier(BachelierParams),
    /// Displaced log-normal (Rubinstein 1983) — shifted Black-76.
    Displaced(DisplacedParams),
}

impl Pricing for Model {
    #[inline(always)]
    fn price(&self) -> Result<f64, PricingError> {
        match self {
            Self::BlackScholes(p) => p.price(),
            Self::Black76(p) => p.price(),
            Self::Bachelier(p) => p.price(),
            Self::Displaced(p) => p.price(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_type_clone_copy() {
        let call = OptionType::Call;
        let call2 = call;
        assert_eq!(call, call2);

        let put = OptionType::Put;
        assert_ne!(call, put);
    }

    #[test]
    fn test_option_type_debug() {
        let call = OptionType::Call;
        let debug_str = format!("{call:?}");
        assert_eq!(debug_str, "Call");
    }

    #[test]
    fn test_option_params_f64_construction() {
        let params = OptionParams {
            option_type: OptionType::Call,
            spot: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        assert_eq!(params.spot.to_f64(), 100.0_f64);
        assert_eq!(params.vol.to_f64(), 0.20_f64);
    }

    #[test]
    fn test_option_params_f32_construction() {
        let params = OptionParams {
            option_type: OptionType::Put,
            spot: 100.0_f32,
            strike: 110.0_f32,
            rate: 0.05_f32,
            div_yield: 0.02_f32,
            vol: 0.20_f32,
            time: 1.0_f32,
        };
        assert!((params.spot.to_f64() - 100.0_f64).abs() < 1e-5_f64);
    }

    #[test]
    fn test_option_params_copy() {
        let p1 = OptionParams {
            option_type: OptionType::Call,
            spot: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p2 = p1;
        assert_eq!(p1.spot.to_f64(), p2.spot.to_f64());
    }

    #[test]
    fn test_greeks_f64_construction() {
        let g = Greeks {
            delta: 0.5987_f64,
            gamma: 0.0185_f64,
            theta: -0.0152_f64,
            vega: 0.3702_f64,
            rho: 0.4174_f64,
            epsilon: -0.5702_f64,
            lambda: 6.47_f64,
            vanna: -0.1314_f64,
            charm: -0.0265_f64,
            veta: 0.0_f64,
            vomma: 0.1499_f64,
            speed: -0.0006_f64,
            zomma: -0.0009_f64,
            color: 0.0_f64,
            ultima: 0.0_f64,
            dual_delta: -0.4741_f64,
            dual_gamma: 0.0_f64,
        };
        assert!(g.delta > 0.0_f64);
        assert!(g.gamma > 0.0_f64);
        assert!(g.theta < 0.0_f64);
    }

    #[test]
    fn test_greeks_copy() {
        let g1 = Greeks {
            delta: 0.5_f64,
            gamma: 0.01_f64,
            theta: -0.01_f64,
            vega: 0.3_f64,
            rho: 0.4_f64,
            epsilon: -0.5_f64,
            lambda: 6.0_f64,
            vanna: -0.1_f64,
            charm: -0.02_f64,
            veta: 0.0_f64,
            vomma: 0.1_f64,
            speed: -0.0006_f64,
            zomma: -0.0009_f64,
            color: 0.0_f64,
            ultima: 0.0_f64,
            dual_delta: -0.4_f64,
            dual_gamma: 0.0_f64,
        };
        let g2 = g1;
        assert_eq!(g1.delta.to_f64(), g2.delta.to_f64());
    }

    #[test]
    fn test_float_f64_zero_one() {
        assert_eq!(f64::zero(), 0.0_f64);
        assert_eq!(f64::one(), 1.0_f64);
    }

    #[test]
    fn test_float_f64_ln_exp() {
        let val = 2.0_f64;
        let result = val.ln().exp();
        assert!((result - 2.0_f64).abs() < 1e-15_f64);
    }

    #[test]
    fn test_float_f64_sqrt() {
        let val = 4.0_f64;
        assert!((val.sqrt() - 2.0_f64).abs() < 1e-15_f64);
    }

    #[test]
    fn test_float_f64_abs() {
        assert_eq!((-3.0_f64).abs(), 3.0_f64);
        assert_eq!(3.0_f64.abs(), 3.0_f64);
    }

    #[test]
    fn test_float_f64_mul_add() {
        // 2.0 * 3.0 + 4.0 = 10.0
        let result = 2.0_f64.mul_add(3.0_f64, 4.0_f64);
        assert!((result - 10.0_f64).abs() < 1e-15_f64);
    }

    #[test]
    fn test_float_f64_from_f64() {
        let val = f64::from_f64(3.14_f64);
        assert!((val - 3.14_f64).abs() < 1e-15_f64);
    }

    #[test]
    fn test_float_f32_roundtrip() {
        let val = f32::from_f64(3.14_f64);
        let back = val.to_f64();
        assert!((back - 3.14_f64).abs() < 1e-5_f64);
    }

    #[test]
    fn test_float_f64_pi() {
        assert!((f64::pi() - core::f64::consts::PI).abs() < 1e-15_f64);
    }

    #[test]
    fn test_float_f64_is_nan() {
        assert!(f64::NAN.is_nan());
        assert!(!1.0_f64.is_nan());
    }

    #[test]
    fn test_float_f64_is_infinite() {
        assert!(f64::INFINITY.is_infinite());
        assert!(!1.0_f64.is_infinite());
    }

    #[test]
    fn test_float_f32_basic_ops() {
        let a = 2.0_f32;
        let b = 3.0_f32;
        assert!((a + b - 5.0_f32).abs() < 1e-6_f32);
        assert!((a * b - 6.0_f32).abs() < 1e-6_f32);
        assert!((b - a - 1.0_f32).abs() < 1e-6_f32);
        assert!((b / a - 1.5_f32).abs() < 1e-6_f32);
        assert!((-a + 2.0_f32).abs() < 1e-6_f32);
    }

    // ── Pricing trait tests ────────────────────────────────────────────

    #[test]
    fn test_pricing_trait_bs_call() {
        let params = OptionParams {
            option_type: OptionType::Call,
            spot: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p = params.price().unwrap();
        assert!((p - 9.2270_f64).abs() < 1e-4_f64, "BS call via trait: got {p}");
    }

    #[test]
    fn test_pricing_trait_black76() {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p = params.price().unwrap();
        assert!(p > 0.0_f64, "Black76 via trait: got {p}");
    }

    #[test]
    fn test_pricing_trait_bachelier() {
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        let p = params.price().unwrap();
        assert!(p > 0.0_f64, "Bachelier via trait: got {p}");
    }

    #[test]
    fn test_pricing_trait_displaced() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 50.0_f64,
        };
        let p = params.price().unwrap();
        assert!(p > 0.0_f64, "Displaced via trait: got {p}");
    }

    // ── GreeksCalc trait tests ─────────────────────────────────────────

    #[test]
    fn test_greeks_calc_trait_call() {
        let params = OptionParams {
            option_type: OptionType::Call,
            spot: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let g = params.greeks().unwrap();
        assert!(g.delta > 0.0_f64 && g.delta < 1.0_f64);
        assert!(g.gamma > 0.0_f64);
        assert!(g.vega > 0.0_f64);
    }

    // ── ImpliedVol trait tests ─────────────────────────────────────────

    #[test]
    fn test_implied_vol_trait_roundtrip() {
        let params = OptionParams {
            option_type: OptionType::Call,
            spot: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let market_price = params.price().unwrap();
        let iv_params = OptionParams {
            vol: 0.0_f64,
            ..params
        };
        let iv = iv_params.implied_vol(market_price, IvSolver::Auto).unwrap();
        assert!(
            (iv - 0.20_f64).abs() < 1e-6_f64,
            "IV roundtrip: expected ~0.20, got {iv}"
        );
    }

    // ── Model enum tests ───────────────────────────────────────────────

    #[test]
    fn test_model_enum_bs() {
        let m = Model::BlackScholes(OptionParams {
            option_type: OptionType::Call,
            spot: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        });
        let p = m.price().unwrap();
        assert!((p - 9.2270_f64).abs() < 1e-4_f64, "Model::BS: got {p}");
    }

    #[test]
    fn test_model_enum_black76() {
        let m = Model::Black76(Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        });
        let p = m.price().unwrap();
        assert!(p > 0.0_f64, "Model::Black76: got {p}");
    }

    #[test]
    fn test_model_enum_bachelier() {
        let m = Model::Bachelier(BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        });
        let p = m.price().unwrap();
        assert!(p > 0.0_f64, "Model::Bachelier: got {p}");
    }

    #[test]
    fn test_model_enum_displaced() {
        let m = Model::Displaced(DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 50.0_f64,
        });
        let p = m.price().unwrap();
        assert!(p > 0.0_f64, "Model::Displaced: got {p}");
    }

    #[test]
    fn test_model_enum_matches_direct_call() {
        let params = OptionParams {
            option_type: OptionType::Call,
            spot: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let direct = params.price().unwrap();
        let via_model = Model::BlackScholes(params).price().unwrap();
        assert!(
            (direct - via_model).abs() < 1e-15_f64,
            "Model dispatch must match direct call"
        );
    }
}
