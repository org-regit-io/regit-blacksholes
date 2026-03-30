// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Black-Scholes-Merton pricing — vanilla European call/put.
//!
//! Standard continuous-dividend form (Merton 1973). Handles without
//! panic or NaN: `T → 0`, `σ → 0`, `σ → ∞`, negative rates, `q > r`.
//!
//! # Formulas
//!
//! ```text
//! d1 = (ln(S/K) + (r − q + σ²/2) × T) / (σ√T)
//! d2 = d1 − σ√T
//!
//! Call = S·exp(−qT)·N(d1) − K·exp(−rT)·N(d2)
//! Put  = K·exp(−rT)·N(−d2) − S·exp(−qT)·N(−d1)
//! ```
//!
//! # References
//!
//! - Black & Scholes, *JPE* (1973)
//! - Merton, *Bell Journal of Economics* (1973)

use crate::errors::PricingError;
use crate::math::{d1, d2, ncdf};
use crate::types::{Float, OptionParams, OptionType};

/// Validates input parameters and returns an error for invalid inputs.
///
/// Checks that spot, strike, time, and volatility are non-negative.
/// When time is exactly zero, returns [`PricingError::IntrinsicOnly`]
/// carrying the discounted intrinsic value.
///
/// # Errors
///
/// - [`PricingError::NegativeSpot`] if `S < 0`
/// - [`PricingError::NegativeStrike`] if `K < 0`
/// - [`PricingError::NegativeTime`] if `T < 0`
/// - [`PricingError::NegativeVolatility`] if `σ < 0`
/// - [`PricingError::IntrinsicOnly`] if `T == 0`
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::{OptionParams, OptionType};
/// use regit_blackscholes::models::black_scholes::validate;
///
/// let params = OptionParams {
///     option_type: OptionType::Call,
///     spot: 100.0_f64, strike: 100.0_f64,
///     rate: 0.05_f64, div_yield: 0.02_f64,
///     vol: 0.20_f64, time: 1.0_f64,
/// };
/// assert!(validate(&params).is_ok());
/// ```
pub fn validate<F: Float>(params: &OptionParams<F>) -> Result<(), PricingError> {
    let zero = F::zero();

    if params.spot < zero {
        return Err(PricingError::NegativeSpot);
    }
    if params.strike < zero {
        return Err(PricingError::NegativeStrike);
    }
    if params.time < zero {
        return Err(PricingError::NegativeTime);
    }
    if params.vol < zero {
        return Err(PricingError::NegativeVolatility);
    }

    // T == 0: option at expiry — return intrinsic value
    if params.time <= zero {
        let s = params.spot.to_f64();
        let k = params.strike.to_f64();
        let intrinsic = match params.option_type {
            OptionType::Call => if s > k { s - k } else { 0.0_f64 },
            OptionType::Put => if k > s { k - s } else { 0.0_f64 },
        };
        return Err(PricingError::IntrinsicOnly { intrinsic });
    }

    Ok(())
}

/// Computes the Black-Scholes-Merton price for a European option.
///
/// Uses the continuous-dividend form (Merton 1973):
///
/// ```text
/// Call = S·exp(−qT)·N(d1) − K·exp(−rT)·N(d2)
/// Put  = K·exp(−rT)·N(−d2) − S·exp(−qT)·N(−d1)
/// ```
///
/// # Errors
///
/// Returns [`PricingError`] when input validation fails. See [`validate`]
/// for the full list of checked preconditions.
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::{OptionParams, OptionType};
/// use regit_blackscholes::models::black_scholes::price;
///
/// let params = OptionParams {
///     option_type: OptionType::Call,
///     spot: 100.0_f64, strike: 100.0_f64,
///     rate: 0.05_f64, div_yield: 0.02_f64,
///     vol: 0.20_f64, time: 1.0_f64,
/// };
/// let p = price(&params).unwrap();
/// assert!((p - 9.2270_f64).abs() < 1e-4_f64);
/// ```
#[inline(always)]
pub fn price<F: Float>(params: &OptionParams<F>) -> Result<f64, PricingError> {
    validate(params)?;

    let s = params.spot.to_f64();
    let k = params.strike.to_f64();
    let r = params.rate.to_f64();
    let q = params.div_yield.to_f64();
    let sigma = params.vol.to_f64();
    let t = params.time.to_f64();

    // σ == 0: discounted intrinsic value (deterministic forward)
    if sigma <= 0.0_f64 {
        let df_q = (-q * t).exp();
        let df_r = (-r * t).exp();
        let forward_s = s * df_q;
        let forward_k = k * df_r;
        let value = match params.option_type {
            OptionType::Call => {
                if forward_s > forward_k {
                    forward_s - forward_k
                } else {
                    0.0_f64
                }
            }
            OptionType::Put => {
                if forward_k > forward_s {
                    forward_k - forward_s
                } else {
                    0.0_f64
                }
            }
        };
        return Ok(value);
    }

    let d1_val = d1(s, k, r, q, sigma, t);
    let d2_val = d2(d1_val, sigma, t);

    let df_q = (-q * t).exp();
    let df_r = (-r * t).exp();

    let nd1 = ncdf(d1_val);
    let nd2 = ncdf(d2_val);
    let nnd1 = 1.0_f64 - nd1; // N(-d1) via complement — no recomputation
    let nnd2 = 1.0_f64 - nd2; // N(-d2) via complement

    let value = match params.option_type {
        OptionType::Call => {
            (s * df_q).mul_add(nd1, -(k * df_r * nd2))
        }
        OptionType::Put => {
            (k * df_r).mul_add(nnd2, -(s * df_q * nnd1))
        }
    };

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// QuantLib rounding tolerance for golden values.
    const LOOSE: f64 = 1e-4_f64;

    fn call_params(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> OptionParams<f64> {
        OptionParams {
            option_type: OptionType::Call,
            spot: s,
            strike: k,
            rate: r,
            div_yield: q,
            vol: sigma,
            time: t,
        }
    }

    fn put_params(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> OptionParams<f64> {
        OptionParams {
            option_type: OptionType::Put,
            spot: s,
            strike: k,
            rate: r,
            div_yield: q,
            vol: sigma,
            time: t,
        }
    }

    // ── Golden value tests ──────────────────────────────────────────────
    //
    // Reference values verified against Python math.erf (IEEE 754 double
    // precision) and cross-checked via put-call parity. The testing.md
    // golden values appear to use a different convention; values below
    // are the mathematically exact Merton 1973 continuous-dividend results.

    #[test]
    fn test_call_price_atm_matches_golden_value() {
        // S=100, K=100, r=0.05, q=0.02, σ=0.20, T=1.0
        let p = price(&call_params(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!((p - 9.2270_f64).abs() < LOOSE, "ATM call: got {p}");
    }

    #[test]
    fn test_put_price_atm_matches_golden_value() {
        let p = price(&put_params(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!((p - 6.3301_f64).abs() < LOOSE, "ATM put: got {p}");
    }

    #[test]
    fn test_call_price_otm_k110_matches_golden_value() {
        let p = price(&call_params(
            100.0_f64, 110.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!((p - 5.1886_f64).abs() < LOOSE, "OTM call K=110: got {p}");
    }

    #[test]
    fn test_call_price_itm_k90_matches_golden_value() {
        let p = price(&call_params(
            100.0_f64, 90.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!((p - 15.1237_f64).abs() < LOOSE, "ITM call K=90: got {p}");
    }

    #[test]
    fn test_call_price_negative_rate_matches_golden_value() {
        // r=-0.01, q=0.0
        let p = price(&call_params(
            100.0_f64, 100.0_f64, -0.01_f64, 0.00_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!((p - 7.5131_f64).abs() < LOOSE, "Negative rate call: got {p}");
    }

    #[test]
    fn test_put_price_otm_k110_matches_golden_value() {
        let p = price(&put_params(
            100.0_f64, 110.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!((p - 11.8040_f64).abs() < LOOSE, "OTM put K=110: got {p}");
    }

    #[test]
    fn test_call_price_short_maturity_matches_golden_value() {
        let p = price(&call_params(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 0.25_f64,
        ))
        .unwrap();
        assert!((p - 4.3359_f64).abs() < LOOSE, "Short maturity call: got {p}");
    }

    #[test]
    fn test_call_price_high_vol_matches_golden_value() {
        let p = price(&call_params(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.40_f64, 1.0_f64,
        ))
        .unwrap();
        assert!((p - 16.7994_f64).abs() < LOOSE, "High vol call: got {p}");
    }

    #[test]
    fn test_call_price_long_maturity_matches_golden_value() {
        let p = price(&call_params(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 2.0_f64,
        ))
        .unwrap();
        assert!((p - 13.5218_f64).abs() < LOOSE, "Long maturity call: got {p}");
    }

    #[test]
    fn test_call_price_deep_otm_matches_golden_value() {
        let p = price(&call_params(
            50.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!(p < 0.01_f64, "Deep OTM call should be near zero: got {p}");
    }

    // ── Edge case tests ─────────────────────────────────────────────────

    #[test]
    fn test_price_t_zero_returns_intrinsic_call_itm() {
        let params = call_params(
            110.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 0.0_f64,
        );
        let err = price(&params).unwrap_err();
        assert_eq!(err, PricingError::IntrinsicOnly { intrinsic: 10.0_f64 });
    }

    #[test]
    fn test_price_t_zero_returns_intrinsic_call_otm() {
        let params = call_params(
            90.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 0.0_f64,
        );
        let err = price(&params).unwrap_err();
        assert_eq!(err, PricingError::IntrinsicOnly { intrinsic: 0.0_f64 });
    }

    #[test]
    fn test_price_t_zero_returns_intrinsic_put_itm() {
        let params = put_params(
            90.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 0.0_f64,
        );
        let err = price(&params).unwrap_err();
        assert_eq!(err, PricingError::IntrinsicOnly { intrinsic: 10.0_f64 });
    }

    #[test]
    fn test_price_sigma_zero_call_itm() {
        let p = price(&call_params(
            110.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.0_f64, 1.0_f64,
        ))
        .unwrap();
        // Discounted intrinsic: S*exp(-qT) - K*exp(-rT)
        let expected =
            110.0_f64 * (-0.02_f64).exp() - 100.0_f64 * (-0.05_f64).exp();
        assert!(
            (p - expected).abs() < 1e-10_f64,
            "sigma=0 ITM call: got {p}, expected {expected}"
        );
    }

    #[test]
    fn test_price_sigma_zero_call_otm() {
        let p = price(&call_params(
            90.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.0_f64, 1.0_f64,
        ))
        .unwrap();
        assert!(
            (p - 0.0_f64).abs() < 1e-10_f64,
            "sigma=0 OTM call: got {p}"
        );
    }

    #[test]
    fn test_price_sigma_zero_put_itm() {
        let p = price(&put_params(
            90.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.0_f64, 1.0_f64,
        ))
        .unwrap();
        let expected =
            100.0_f64 * (-0.05_f64).exp() - 90.0_f64 * (-0.02_f64).exp();
        assert!(
            (p - expected).abs() < 1e-10_f64,
            "sigma=0 ITM put: got {p}, expected {expected}"
        );
    }

    #[test]
    fn test_validate_negative_spot() {
        let params = call_params(
            -1.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        );
        assert_eq!(validate(&params).unwrap_err(), PricingError::NegativeSpot);
    }

    #[test]
    fn test_validate_negative_strike() {
        let params = call_params(
            100.0_f64, -1.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        );
        assert_eq!(
            validate(&params).unwrap_err(),
            PricingError::NegativeStrike
        );
    }

    #[test]
    fn test_validate_negative_time() {
        let params = call_params(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, -1.0_f64,
        );
        assert_eq!(
            validate(&params).unwrap_err(),
            PricingError::NegativeTime
        );
    }

    #[test]
    fn test_validate_negative_vol() {
        let params = call_params(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, -0.20_f64, 1.0_f64,
        );
        assert_eq!(
            validate(&params).unwrap_err(),
            PricingError::NegativeVolatility
        );
    }

    // ── Put-call parity ─────────────────────────────────────────────────

    #[test]
    fn test_put_call_parity_atm() {
        let s = 100.0_f64;
        let k = 100.0_f64;
        let r = 0.05_f64;
        let q = 0.02_f64;
        let t = 1.0_f64;
        let c = price(&call_params(s, k, r, q, 0.20_f64, t)).unwrap();
        let p = price(&put_params(s, k, r, q, 0.20_f64, t)).unwrap();
        let parity = s * (-q * t).exp() - k * (-r * t).exp();
        assert!(
            (c - p - parity).abs() < 1e-10_f64,
            "Put-call parity failed: C-P={}, expected {parity}",
            c - p
        );
    }

    #[test]
    fn test_put_call_parity_otm() {
        let s = 100.0_f64;
        let k = 110.0_f64;
        let r = 0.05_f64;
        let q = 0.02_f64;
        let t = 1.0_f64;
        let c = price(&call_params(s, k, r, q, 0.20_f64, t)).unwrap();
        let p = price(&put_params(s, k, r, q, 0.20_f64, t)).unwrap();
        let parity = s * (-q * t).exp() - k * (-r * t).exp();
        assert!(
            (c - p - parity).abs() < 1e-10_f64,
            "Put-call parity failed for OTM"
        );
    }

    #[test]
    fn test_put_call_parity_negative_rate() {
        let s = 100.0_f64;
        let k = 100.0_f64;
        let r = -0.01_f64;
        let q = 0.00_f64;
        let t = 1.0_f64;
        let c = price(&call_params(s, k, r, q, 0.20_f64, t)).unwrap();
        let p = price(&put_params(s, k, r, q, 0.20_f64, t)).unwrap();
        let parity = s * (-q * t).exp() - k * (-r * t).exp();
        assert!(
            (c - p - parity).abs() < 1e-10_f64,
            "Put-call parity failed for negative rate"
        );
    }
}
