// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Bachelier (Normal) pricing model.
//!
//! Assumes normally distributed returns (not log-normal). Required for
//! rates options near or below zero. Vol parameter is absolute (normal
//! vol in price units), not relative.
//!
//! # Formulas
//!
//! ```text
//! d    = (F - K) / (sigma_N * sqrt(T))
//! Call = exp(-rT) * [(F - K) * N(d) + sigma_N * sqrt(T) * phi(d)]
//! Put  = exp(-rT) * [(K - F) * N(-d) + sigma_N * sqrt(T) * phi(d)]
//! ```
//!
//! where `phi(d)` is the standard normal PDF and `N(d)` is the standard
//! normal CDF.
//!
//! # References
//!
//! - Bachelier, L., *Theorie de la Speculation* (1900)
//! - Schachermayer, W. & Teichmann, J. (2008)

use crate::errors::PricingError;
use crate::math::{ncdf, npdf};
use crate::types::OptionType;

/// Parameters for Bachelier (normal model) pricing.
///
/// # Fields
///
/// | Field | Symbol | Description |
/// |-------|--------|-------------|
/// | `forward` | F | Forward price |
/// | `strike` | K | Strike price |
/// | `rate` | r | Risk-free rate (continuous, annualised) |
/// | `normal_vol` | sigma_N | Normal volatility in price units (NOT percentage) |
/// | `time` | T | Time to expiry in years |
/// | `option_type` | -- | Call or Put |
///
/// # Examples
///
/// ```
/// use regit_blackscholes::models::bachelier::BachelierParams;
/// use regit_blackscholes::types::OptionType;
///
/// let params = BachelierParams {
///     option_type: OptionType::Call,
///     forward:    100.0_f64,
///     strike:     100.0_f64,
///     rate:       0.05_f64,
///     normal_vol: 5.0_f64,
///     time:       1.0_f64,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BachelierParams {
    /// Option type -- call or put.
    pub option_type: OptionType,
    /// F -- forward price.
    pub forward: f64,
    /// K -- strike price.
    pub strike: f64,
    /// r -- risk-free rate (continuous, annualised).
    pub rate: f64,
    /// sigma_N -- normal volatility in price units (NOT percentage).
    pub normal_vol: f64,
    /// T -- time to expiry in years.
    pub time: f64,
}

/// Computes the Bachelier (normal model) price for a European option.
///
/// Returns `exp(-rT) * [(F-K)*N(d) + sigma_N*sqrt(T)*phi(d)]` for calls
/// (analogous for puts), where `d = (F-K)/(sigma_N*sqrt(T))`.
///
/// # Arguments
///
/// * `params` -- Option parameters (forward, strike, rate, normal_vol, time, type).
///
/// # Returns
///
/// The fair value of the option under the normal model, discounted at the
/// risk-free rate.
///
/// # Errors
///
/// Returns [`PricingError`] if any input is invalid:
/// - `time < 0` -> [`PricingError::NegativeTime`]
/// - `normal_vol < 0` -> [`PricingError::NegativeVolatility`]
/// - `time == 0` -> [`PricingError::IntrinsicOnly`] with intrinsic value
///
/// Note: negative forwards and strikes are valid in the Bachelier model
/// (rates markets with negative rates).
///
/// # Examples
///
/// ```
/// use regit_blackscholes::models::bachelier::{BachelierParams, price};
/// use regit_blackscholes::types::OptionType;
///
/// let params = BachelierParams {
///     option_type: OptionType::Call,
///     forward:    100.0_f64,
///     strike:     100.0_f64,
///     rate:       0.05_f64,
///     normal_vol: 5.0_f64,
///     time:       1.0_f64,
/// };
/// let p = price(&params).unwrap();
/// assert!(p > 0.0_f64);
/// ```
#[inline(always)]
pub fn price(params: &BachelierParams) -> Result<f64, PricingError> {
    // ── Input validation ─────────────────────────────────────────────
    // Note: Bachelier model is designed for rates near/below zero.
    // Negative forwards and strikes are valid in rates markets.
    if params.time < 0.0_f64 {
        return Err(PricingError::NegativeTime);
    }
    if params.normal_vol < 0.0_f64 {
        return Err(PricingError::NegativeVolatility);
    }

    let f = params.forward;
    let k = params.strike;
    let r = params.rate;
    let sigma_n = params.normal_vol;
    let t = params.time;

    // ── Edge case: T == 0 -> intrinsic ───────────────────────────────
    if t == 0.0_f64 {
        let intrinsic = match params.option_type {
            OptionType::Call => {
                if f > k { f - k } else { 0.0_f64 }
            }
            OptionType::Put => {
                if k > f { k - f } else { 0.0_f64 }
            }
        };
        return Err(PricingError::IntrinsicOnly { intrinsic });
    }

    // ── Core computation ─────────────────────────────────────────────
    let sqrt_t = t.sqrt();
    let sigma_sqrt_t = sigma_n * sqrt_t;
    let discount = (-r * t).exp();

    // Handle sigma_N == 0: return discounted intrinsic
    if sigma_sqrt_t == 0.0_f64 {
        let intrinsic = match params.option_type {
            OptionType::Call => {
                if f > k { f - k } else { 0.0_f64 }
            }
            OptionType::Put => {
                if k > f { k - f } else { 0.0_f64 }
            }
        };
        return Ok(discount * intrinsic);
    }

    let d = (f - k) / sigma_sqrt_t;
    let phi_d = npdf(d);

    let price_val = match params.option_type {
        OptionType::Call => {
            discount * (f - k).mul_add(ncdf(d), sigma_sqrt_t * phi_d)
        }
        OptionType::Put => {
            discount * (k - f).mul_add(ncdf(-d), sigma_sqrt_t * phi_d)
        }
    };

    Ok(price_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for golden value comparisons -- `LOOSE = 1e-4`.
    const LOOSE: f64 = 1e-4_f64;

    // ── Golden values ────────────────────────────────────────────────
    // Reference: Bachelier formula, exp(-rT)*[(F-K)*N(d) + sigma_N*sqrt(T)*phi(d)].
    // F=100, K=100, r=0.05, sigma_N=5.0, T=1.0

    #[test]
    fn test_call_atm_matches_golden_value() {
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 1.8974_f64).abs() < LOOSE, "call ATM: got {p}");
    }

    #[test]
    fn test_put_atm_matches_golden_value() {
        let params = BachelierParams {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 1.8974_f64).abs() < LOOSE, "put ATM: got {p}");
    }

    #[test]
    fn test_call_otm_matches_golden_value() {
        // F=100, K=105
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 105.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 0.3963_f64).abs() < LOOSE, "call OTM: got {p}");
    }

    #[test]
    fn test_put_otm_matches_golden_value() {
        // F=100, K=105
        let params = BachelierParams {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 105.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 5.1524_f64).abs() < LOOSE, "put OTM: got {p}");
    }

    #[test]
    fn test_call_high_vol_matches_golden_value() {
        // F=100, K=100, sigma_N=10.0
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 10.0_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 3.7949_f64).abs() < LOOSE, "call high vol: got {p}");
    }

    #[test]
    fn test_put_high_vol_matches_golden_value() {
        // F=100, K=100, sigma_N=10.0
        let params = BachelierParams {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 10.0_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 3.7949_f64).abs() < LOOSE, "put high vol: got {p}");
    }

    // ── Put-call parity ──────────────────────────────────────────────

    #[test]
    fn test_put_call_parity_atm() {
        // C - P = exp(-rT) * (F - K) = 0 when F = K
        let call_params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        let put_params = BachelierParams {
            option_type: OptionType::Put,
            ..call_params
        };
        let c = price(&call_params).unwrap();
        let p = price(&put_params).unwrap();
        assert!((c - p).abs() < 1e-10_f64, "parity ATM: C-P={}", c - p);
    }

    #[test]
    fn test_put_call_parity_otm() {
        // C - P = exp(-rT) * (F - K)
        let call_params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 105.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        let put_params = BachelierParams {
            option_type: OptionType::Put,
            ..call_params
        };
        let c = price(&call_params).unwrap();
        let p = price(&put_params).unwrap();
        let parity = (-0.05_f64 * 1.0_f64).exp() * (100.0_f64 - 105.0_f64);
        assert!((c - p - parity).abs() < 1e-10_f64, "parity: C-P={}, expected={parity}", c - p);
    }

    // ── Input validation ─────────────────────────────────────────────

    #[test]
    fn test_negative_forward_is_valid() {
        // Bachelier model supports negative forwards (rates markets)
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: -1.0_f64,
            strike: 0.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        assert!(price(&params).is_ok());
    }

    #[test]
    fn test_negative_strike_is_valid() {
        // Bachelier supports negative strikes (rates markets)
        let params = BachelierParams {
            option_type: OptionType::Put,
            forward: 0.0_f64,
            strike: -1.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        assert!(price(&params).is_ok());
    }

    #[test]
    fn test_negative_time_returns_error() {
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: -1.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeTime));
    }

    #[test]
    fn test_negative_vol_returns_error() {
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: -5.0_f64,
            time: 1.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeVolatility));
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_t_zero_returns_intrinsic_call_itm() {
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 0.0_f64,
        };
        match price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert!((intrinsic - 10.0_f64).abs() < 1e-10_f64);
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_t_zero_returns_intrinsic_put_otm() {
        let params = BachelierParams {
            option_type: OptionType::Put,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 0.0_f64,
        };
        match price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert!((intrinsic - 0.0_f64).abs() < 1e-10_f64);
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_zero_vol_returns_discounted_intrinsic() {
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 0.0_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        let expected = (-0.05_f64).exp() * 10.0_f64;
        assert!((p - expected).abs() < 1e-10_f64, "zero vol: got {p}, expected {expected}");
    }

    #[test]
    fn test_atm_call_equals_put() {
        let call_params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        let put_params = BachelierParams {
            option_type: OptionType::Put,
            ..call_params
        };
        let c = price(&call_params).unwrap();
        let p = price(&put_params).unwrap();
        assert!((c - p).abs() < 1e-10_f64, "ATM symmetry: C={c}, P={p}");
    }

    #[test]
    fn test_negative_rate_no_panic() {
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: -0.02_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!(p.is_finite(), "negative rate produced non-finite: {p}");
        assert!(p > 0.0_f64);
    }

    #[test]
    fn test_price_proportional_to_vol_atm() {
        // For ATM Bachelier, price is proportional to sigma_N
        let p1 = price(&BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        }).unwrap();
        let p2 = price(&BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 10.0_f64,
            time: 1.0_f64,
        }).unwrap();
        assert!((p2 / p1 - 2.0_f64).abs() < 1e-10_f64, "ratio: {}", p2 / p1);
    }
}
