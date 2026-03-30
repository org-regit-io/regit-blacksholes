// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Black-76 pricing -- options on futures and forwards.
//!
//! Eliminates the dividend yield parameter -- the forward `F` is the
//! primary input. Required for interest rate caps/floors, commodity
//! options, swaptions under log-normal vol convention.
//!
//! # Formulas
//!
//! ```text
//! d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt(T))
//! d2 = d1 - sigma sqrt(T)
//!
//! Call = exp(-rT) * [F * N(d1) - K * N(d2)]
//! Put  = exp(-rT) * [K * N(-d2) - F * N(-d1)]
//! ```
//!
//! # References
//!
//! - Black, F., "The pricing of commodity contracts",
//!   *Journal of Financial Economics*, 3(1-2):167-179 (1976)

use crate::errors::PricingError;
use crate::math::ncdf;
use crate::types::OptionType;

/// Parameters for Black-76 futures/forwards pricing.
///
/// # Fields
///
/// | Field | Symbol | Description |
/// |-------|--------|-------------|
/// | `forward` | F | Forward price |
/// | `strike` | K | Strike price |
/// | `rate` | r | Risk-free rate (continuous, annualised) |
/// | `vol` | sigma | Log-normal volatility (annualised) |
/// | `time` | T | Time to expiry in years |
/// | `option_type` | -- | Call or Put |
///
/// # Examples
///
/// ```
/// use regit_blackscholes::models::black76::Black76Params;
/// use regit_blackscholes::types::OptionType;
///
/// let params = Black76Params {
///     option_type: OptionType::Call,
///     forward: 100.0_f64,
///     strike:  100.0_f64,
///     rate:    0.05_f64,
///     vol:     0.20_f64,
///     time:    1.0_f64,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Black76Params {
    /// Option type -- call or put.
    pub option_type: OptionType,
    /// F -- forward price.
    pub forward: f64,
    /// K -- strike price.
    pub strike: f64,
    /// r -- risk-free rate (continuous, annualised).
    pub rate: f64,
    /// sigma -- log-normal volatility (annualised).
    pub vol: f64,
    /// T -- time to expiry in years.
    pub time: f64,
}

/// Computes the Black-76 price for a European option on a future/forward.
///
/// Returns the present value `exp(-rT) * [F*N(d1) - K*N(d2)]` for calls
/// (analogous for puts), where d1 and d2 are computed once from the
/// forward price and vol.
///
/// # Arguments
///
/// * `params` -- Option parameters (forward, strike, rate, vol, time, type).
///
/// # Returns
///
/// The fair value of the option, discounted at the risk-free rate.
///
/// # Errors
///
/// Returns [`PricingError`] if any input is invalid:
/// - `forward < 0` -> [`PricingError::NegativeSpot`]
/// - `strike < 0` -> [`PricingError::NegativeStrike`]
/// - `time < 0` -> [`PricingError::NegativeTime`]
/// - `vol < 0` -> [`PricingError::NegativeVolatility`]
/// - `time == 0` -> [`PricingError::IntrinsicOnly`] with intrinsic value
///
/// # Examples
///
/// ```
/// use regit_blackscholes::models::black76::{Black76Params, price};
/// use regit_blackscholes::types::OptionType;
///
/// let params = Black76Params {
///     option_type: OptionType::Call,
///     forward: 100.0_f64,
///     strike:  100.0_f64,
///     rate:    0.05_f64,
///     vol:     0.20_f64,
///     time:    1.0_f64,
/// };
/// let p = price(&params).unwrap();
/// assert!(p > 0.0_f64);
/// ```
#[inline(always)]
pub fn price(params: &Black76Params) -> Result<f64, PricingError> {
    // ── Input validation ─────────────────────────────────────────────
    if params.forward < 0.0_f64 {
        return Err(PricingError::NegativeSpot);
    }
    if params.strike < 0.0_f64 {
        return Err(PricingError::NegativeStrike);
    }
    if params.time < 0.0_f64 {
        return Err(PricingError::NegativeTime);
    }
    if params.vol < 0.0_f64 {
        return Err(PricingError::NegativeVolatility);
    }

    let f = params.forward;
    let k = params.strike;
    let r = params.rate;
    let sigma = params.vol;
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
    let sigma_sqrt_t = sigma * sqrt_t;
    let discount = (-r * t).exp();

    // Handle sigma == 0: return discounted intrinsic
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

    let d1 = ((f / k).ln() + 0.5_f64 * sigma * sigma * t) / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;

    let nd1 = ncdf(d1);
    let nd2 = ncdf(d2);
    let nnd1 = 1.0_f64 - nd1; // N(-d1) via complement — no recomputation
    let nnd2 = 1.0_f64 - nd2; // N(-d2) via complement

    let price_val = match params.option_type {
        OptionType::Call => discount * f.mul_add(nd1, -k * nd2),
        OptionType::Put => discount * k.mul_add(nnd2, -f * nnd1),
    };

    Ok(price_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for golden value comparisons -- `LOOSE = 1e-4`.
    const LOOSE: f64 = 1e-4_f64;

    // ── Golden values ────────────────────────────────────────────────
    // Reference: Black-76 formula, exp(-rT)*[F*N(d1)-K*N(d2)].
    // F=100, K=100, r=0.05, sigma=0.20, T=1.0

    #[test]
    fn test_call_atm_matches_golden_value() {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 7.5771_f64).abs() < LOOSE, "call ATM: got {p}");
    }

    #[test]
    fn test_put_atm_matches_golden_value() {
        let params = Black76Params {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 7.5771_f64).abs() < LOOSE, "put ATM: got {p}");
    }

    #[test]
    fn test_call_otm_matches_golden_value() {
        // F=100, K=105
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 105.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 5.6176_f64).abs() < LOOSE, "call OTM: got {p}");
    }

    #[test]
    fn test_put_otm_matches_golden_value() {
        // F=100, K=105
        let params = Black76Params {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 105.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 10.3737_f64).abs() < LOOSE, "put OTM: got {p}");
    }

    #[test]
    fn test_call_itm_matches_golden_value() {
        // F=120, K=100
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 120.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 21.0672_f64).abs() < LOOSE, "call ITM: got {p}");
    }

    #[test]
    fn test_put_itm_matches_golden_value() {
        // F=120, K=100
        let params = Black76Params {
            option_type: OptionType::Put,
            forward: 120.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!((p - 2.0426_f64).abs() < LOOSE, "put ITM: got {p}");
    }

    // ── Put-call parity ──────────────────────────────────────────────

    #[test]
    fn test_put_call_parity_atm() {
        // C - P = exp(-rT) * (F - K) = 0 when F = K
        let call_params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let put_params = Black76Params {
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
        let call_params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 105.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let put_params = Black76Params {
            option_type: OptionType::Put,
            ..call_params
        };
        let c = price(&call_params).unwrap();
        let p = price(&put_params).unwrap();
        let parity = (-0.05_f64 * 1.0_f64).exp() * (100.0_f64 - 105.0_f64);
        assert!((c - p - parity).abs() < 1e-10_f64, "parity: C-P={}, expected={parity}", c - p);
    }

    #[test]
    fn test_put_call_parity_itm() {
        let call_params = Black76Params {
            option_type: OptionType::Call,
            forward: 120.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let put_params = Black76Params {
            option_type: OptionType::Put,
            ..call_params
        };
        let c = price(&call_params).unwrap();
        let p = price(&put_params).unwrap();
        let parity = (-0.05_f64 * 1.0_f64).exp() * (120.0_f64 - 100.0_f64);
        assert!((c - p - parity).abs() < 1e-10_f64, "parity: C-P={}, expected={parity}", c - p);
    }

    // ── Input validation ─────────────────────────────────────────────

    #[test]
    fn test_negative_forward_returns_error() {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: -1.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeSpot));
    }

    #[test]
    fn test_negative_strike_returns_error() {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: -1.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeStrike));
    }

    #[test]
    fn test_negative_time_returns_error() {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: -1.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeTime));
    }

    #[test]
    fn test_negative_vol_returns_error() {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: -0.20_f64,
            time: 1.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeVolatility));
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_t_zero_returns_intrinsic_call_itm() {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
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
        let params = Black76Params {
            option_type: OptionType::Put,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
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
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.0_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        let expected = (-0.05_f64).exp() * 10.0_f64;
        assert!((p - expected).abs() < 1e-10_f64, "zero vol: got {p}, expected {expected}");
    }

    #[test]
    fn test_atm_call_equals_put() {
        let call_params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let put_params = Black76Params {
            option_type: OptionType::Put,
            ..call_params
        };
        let c = price(&call_params).unwrap();
        let p = price(&put_params).unwrap();
        assert!((c - p).abs() < 1e-10_f64, "ATM symmetry: C={c}, P={p}");
    }

    #[test]
    fn test_negative_rate_no_panic() {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: -0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        let p = price(&params).unwrap();
        assert!(p.is_finite(), "negative rate produced non-finite: {p}");
        assert!(p > 0.0_f64);
    }
}
