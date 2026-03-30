// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Displaced Log-Normal pricing model.
//!
//! Adds displacement parameter `β` so `(F + β)` is log-normally
//! distributed. Bridges Black-76 (`β = 0`) and Bachelier (`β → ∞`).
//! Required for SABR vol surface calibration workflows.
//!
//! # Formula
//!
//! The displaced diffusion model transforms to Black-76 with shifted inputs:
//!
//! ```text
//! F' = F + β        (shifted forward)
//! K' = K + β        (shifted strike)
//!
//! d1 = (ln(F'/K') + σ²T/2) / (σ√T)
//! d2 = d1 − σ√T
//!
//! Call = exp(−rT) · [F'·N(d1) − K'·N(d2)]
//! Put  = exp(−rT) · [K'·N(−d2) − F'·N(−d1)]
//! ```
//!
//! # References
//!
//! - Rubinstein, *Journal of Finance* (1983)

use crate::errors::PricingError;
use crate::math::{ncdf, npdf};
use crate::types::OptionType;

/// Parameters for Displaced Log-Normal pricing.
///
/// Bridges Black-76 (`β = 0`) and Bachelier (`β → ∞`).
/// Required for SABR vol surface calibration.
///
/// # Fields
///
/// | Field | Symbol | Description |
/// |-------|--------|-------------|
/// | `forward` | F | Forward price |
/// | `strike` | K | Strike price |
/// | `rate` | r | Risk-free rate (continuous, annualised) |
/// | `vol` | σ | Log-normal vol of the shifted process |
/// | `time` | T | Time to expiry in years |
/// | `displacement` | β | Displacement parameter |
/// | `option_type` | — | Call or Put |
///
/// # References
///
/// - Rubinstein, *Journal of Finance* (1983)
#[derive(Debug, Clone, Copy)]
pub struct DisplacedParams {
    /// Option type — call or put.
    pub option_type: OptionType,
    /// F — forward price.
    pub forward: f64,
    /// K — strike price.
    pub strike: f64,
    /// r — risk-free rate (continuous, annualised).
    pub rate: f64,
    /// σ — log-normal volatility of the shifted process (annualised).
    pub vol: f64,
    /// T — time to expiry in years.
    pub time: f64,
    /// β — displacement parameter.
    pub displacement: f64,
}

/// Computes the displaced log-normal option price.
///
/// Transforms to Black-76 with shifted forward and strike:
/// `F' = F + β`, `K' = K + β`, then applies the standard Black-76 formula.
///
/// # Arguments
///
/// * `params` — displaced log-normal parameters
///
/// # Errors
///
/// Returns [`PricingError::NegativeVolatility`] if `σ < 0`.
/// Returns [`PricingError::NegativeTime`] if `T < 0`.
/// Returns [`PricingError::NegativeSpot`] if `F + β ≤ 0` (shifted forward non-positive).
/// Returns [`PricingError::NegativeStrike`] if `K + β ≤ 0` (shifted strike non-positive).
/// Returns [`PricingError::IntrinsicOnly`] if `T == 0` or `σ == 0`.
///
/// # Examples
///
/// ```
/// use regit_blackscholes::models::displaced::{price, DisplacedParams};
/// use regit_blackscholes::types::OptionType;
///
/// let params = DisplacedParams {
///     option_type: OptionType::Call,
///     forward:     100.0_f64,
///     strike:      100.0_f64,
///     rate:        0.05_f64,
///     vol:         0.20_f64,
///     time:        1.0_f64,
///     displacement: 50.0_f64,
/// };
/// let p = price(&params).unwrap();
/// assert!(p > 0.0_f64);
/// ```
#[inline(always)]
pub fn price(params: &DisplacedParams) -> Result<f64, PricingError> {
    // ── Input validation ────────────────────────────────────────────────
    if params.vol < 0.0_f64 {
        return Err(PricingError::NegativeVolatility);
    }
    if params.time < 0.0_f64 {
        return Err(PricingError::NegativeTime);
    }

    let shifted_forward = params.forward + params.displacement;
    let shifted_strike = params.strike + params.displacement;

    if shifted_forward <= 0.0_f64 {
        return Err(PricingError::NegativeSpot);
    }
    if shifted_strike <= 0.0_f64 {
        return Err(PricingError::NegativeStrike);
    }

    // ── Degenerate edge cases ─────────────────────────────────────────
    let discount = (-params.rate * params.time).exp();

    // T == 0: option at expiry — return intrinsic as error
    if params.time == 0.0_f64 {
        let intrinsic = match params.option_type {
            OptionType::Call => {
                if params.forward > params.strike {
                    discount * (params.forward - params.strike)
                } else {
                    0.0_f64
                }
            }
            OptionType::Put => {
                if params.strike > params.forward {
                    discount * (params.strike - params.forward)
                } else {
                    0.0_f64
                }
            }
        };
        return Err(PricingError::IntrinsicOnly { intrinsic });
    }

    // σ == 0 with T > 0: deterministic forward — return discounted intrinsic as Ok
    if params.vol == 0.0_f64 {
        let intrinsic = match params.option_type {
            OptionType::Call => {
                if params.forward > params.strike {
                    discount * (params.forward - params.strike)
                } else {
                    0.0_f64
                }
            }
            OptionType::Put => {
                if params.strike > params.forward {
                    discount * (params.strike - params.forward)
                } else {
                    0.0_f64
                }
            }
        };
        return Ok(intrinsic);
    }

    // ── Black-76 on shifted inputs ──────────────────────────────────────
    let vol_sqrt_t = params.vol * params.time.sqrt();
    let d1 = ((shifted_forward / shifted_strike).ln()
        + 0.5_f64 * params.vol * params.vol * params.time)
        / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;

    let price = match params.option_type {
        OptionType::Call => {
            discount * shifted_forward.mul_add(ncdf(d1), -shifted_strike * ncdf(d2))
        }
        OptionType::Put => {
            discount * shifted_strike.mul_add(ncdf(-d2), -shifted_forward * ncdf(-d1))
        }
    };

    Ok(price)
}

/// Standard normal PDF — used internally for Greeks (future extension).
///
/// Re-exported from [`crate::math::npdf`].
#[allow(dead_code)]
#[inline(always)]
fn shifted_npdf(d: f64) -> f64 {
    npdf(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loose tolerance for smoke tests.
    const LOOSE: f64 = 1e-4_f64;

    /// Tight tolerance for β = 0 ↔ direct Black-76.
    const TIGHT: f64 = 1e-10_f64;

    /// Helper: direct Black-76 price (no displacement) for comparison.
    /// Implements Black-76 inline to avoid dependency on incomplete module.
    fn black76_price(
        option_type: OptionType,
        forward: f64,
        strike: f64,
        rate: f64,
        vol: f64,
        time: f64,
    ) -> f64 {
        let discount = (-rate * time).exp();
        let vol_sqrt_t = vol * time.sqrt();
        let d1 = ((forward / strike).ln() + 0.5_f64 * vol * vol * time) / vol_sqrt_t;
        let d2 = d1 - vol_sqrt_t;
        match option_type {
            OptionType::Call => {
                discount * forward.mul_add(ncdf(d1), -strike * ncdf(d2))
            }
            OptionType::Put => {
                discount * strike.mul_add(ncdf(-d2), -forward * ncdf(-d1))
            }
        }
    }

    // ── β = 0 should match Black-76 exactly ─────────────────────────────

    #[test]
    fn test_beta_zero_call_matches_black76() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 0.0_f64,
        };
        let displaced = price(&params).unwrap();
        let reference = black76_price(
            OptionType::Call,
            100.0_f64,
            100.0_f64,
            0.05_f64,
            0.20_f64,
            1.0_f64,
        );
        assert!(
            (displaced - reference).abs() < TIGHT,
            "displaced={displaced}, reference={reference}"
        );
    }

    #[test]
    fn test_beta_zero_put_matches_black76() {
        let params = DisplacedParams {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 0.0_f64,
        };
        let displaced = price(&params).unwrap();
        let reference = black76_price(
            OptionType::Put,
            100.0_f64,
            100.0_f64,
            0.05_f64,
            0.20_f64,
            1.0_f64,
        );
        assert!(
            (displaced - reference).abs() < TIGHT,
            "displaced={displaced}, reference={reference}"
        );
    }

    #[test]
    fn test_beta_zero_itm_call() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 120.0_f64,
            strike: 100.0_f64,
            rate: 0.03_f64,
            vol: 0.25_f64,
            time: 0.5_f64,
            displacement: 0.0_f64,
        };
        let displaced = price(&params).unwrap();
        let reference = black76_price(
            OptionType::Call,
            120.0_f64,
            100.0_f64,
            0.03_f64,
            0.25_f64,
            0.5_f64,
        );
        assert!(
            (displaced - reference).abs() < TIGHT,
            "displaced={displaced}, reference={reference}"
        );
    }

    // ── Smoke tests with displacement ───────────────────────────────────

    #[test]
    fn test_positive_displacement_call() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 50.0_f64,
        };
        let p = price(&params).unwrap();
        // With β > 0, ATM displaced call should still be positive
        assert!(p > 0.0_f64, "price should be positive, got {p}");
        // ATM call with displacement should differ from β=0 case
        let p_zero = price(&DisplacedParams {
            displacement: 0.0_f64,
            ..params
        })
        .unwrap();
        assert!(
            (p - p_zero).abs() > LOOSE,
            "displacement should change price"
        );
    }

    #[test]
    fn test_positive_displacement_put() {
        let params = DisplacedParams {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 105.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 50.0_f64,
        };
        let p = price(&params).unwrap();
        assert!(p > 0.0_f64, "put price should be positive, got {p}");
    }

    #[test]
    fn test_put_call_parity_displaced() {
        // Put-call parity: C - P = exp(-rT) * (F - K)
        let call_params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 95.0_f64,
            rate: 0.05_f64,
            vol: 0.25_f64,
            time: 1.0_f64,
            displacement: 30.0_f64,
        };
        let put_params = DisplacedParams {
            option_type: OptionType::Put,
            ..call_params
        };
        let c = price(&call_params).unwrap();
        let p = price(&put_params).unwrap();
        let discount = (-0.05_f64 * 1.0_f64).exp();
        let parity = discount * (100.0_f64 - 95.0_f64);
        assert!(
            (c - p - parity).abs() < LOOSE,
            "put-call parity violated: C-P={}, parity={parity}",
            c - p
        );
    }

    // ── Validation errors ───────────────────────────────────────────────

    #[test]
    fn test_negative_vol_error() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: -0.20_f64,
            time: 1.0_f64,
            displacement: 0.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeVolatility));
    }

    #[test]
    fn test_negative_time_error() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: -1.0_f64,
            displacement: 0.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeTime));
    }

    #[test]
    fn test_negative_shifted_forward_error() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 10.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: -20.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeSpot));
    }

    #[test]
    fn test_negative_shifted_strike_error() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 10.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: -20.0_f64,
        };
        assert_eq!(price(&params), Err(PricingError::NegativeStrike));
    }

    #[test]
    fn test_zero_time_intrinsic_call_itm() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 0.0_f64,
            displacement: 10.0_f64,
        };
        match price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert!(
                    (intrinsic - 10.0_f64).abs() < LOOSE,
                    "expected intrinsic ~10.0, got {intrinsic}"
                );
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_zero_time_intrinsic_put_otm() {
        let params = DisplacedParams {
            option_type: OptionType::Put,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 0.0_f64,
            displacement: 10.0_f64,
        };
        match price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert!(
                    intrinsic.abs() < LOOSE,
                    "expected intrinsic ~0.0, got {intrinsic}"
                );
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_zero_vol_intrinsic() {
        // σ=0 with T>0 returns Ok(discounted intrinsic), not an error
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.0_f64,
            time: 1.0_f64,
            displacement: 10.0_f64,
        };
        match price(&params) {
            Ok(p) => {
                let discount = (-0.05_f64).exp();
                let expected = discount * 10.0_f64;
                assert!(
                    (p - expected).abs() < LOOSE,
                    "expected intrinsic ~{expected}, got {p}"
                );
            }
            other => panic!("expected Ok(intrinsic), got {other:?}"),
        }
    }

    #[test]
    fn test_large_displacement() {
        // Large β should produce a price — no overflow or NaN
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 1000.0_f64,
        };
        let p = price(&params).unwrap();
        assert!(p.is_finite(), "price should be finite, got {p}");
        assert!(p > 0.0_f64, "price should be positive, got {p}");
    }

    #[test]
    fn test_negative_rate() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: -0.01_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 20.0_f64,
        };
        let p = price(&params).unwrap();
        assert!(p.is_finite(), "price should be finite with negative rate");
        assert!(p > 0.0_f64, "ATM call should be positive");
    }
}
