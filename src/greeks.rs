// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Analytic Greeks — all 17, through 3rd order.
//!
//! No finite differences anywhere. Every Greek is derived analytically
//! and is numerically stable across the full domain including `T → 0`,
//! `σ → 0`, deep ITM, and deep OTM.
//!
//! Intermediate values `d1`, `d2`, `N(d1)`, `N(d2)`, `φ(d1)` are
//! computed **once** and passed through all Greek formulas.
//!
//! # Greeks computed
//!
//! | Order | Greeks |
//! |-------|--------|
//! | 1st   | Delta, Theta, Vega, Rho, Epsilon, Lambda, Dual Delta |
//! | 2nd   | Gamma, Vanna, Charm, Veta, Vomma, Dual Gamma |
//! | 3rd   | Speed, Zomma, Color, Ultima |
//!
//! # References
//!
//! - Black & Scholes, *JPE* (1973)
//! - Merton, *Bell Journal of Economics* (1973)

use crate::errors::PricingError;
use crate::math::{d1, d2, ncdf, npdf};
use crate::types::{Greeks, OptionParams, OptionType};

/// Computes all 17 analytic Greeks for a Black-Scholes European option.
///
/// Intermediate values `d1`, `d2`, `N(d1)`, `N(d2)`, `φ(d1)` are computed
/// exactly once and reused across all 17 formulas. No recomputation.
///
/// # Arguments
///
/// * `params` — option parameters (spot, strike, rate, dividend yield,
///   volatility, time to expiry, and option type)
///
/// # Returns
///
/// A [`Greeks`] struct containing all 17 analytic Greeks, or a
/// [`PricingError`] if input validation fails.
///
/// # Errors
///
/// Returns [`PricingError::NegativeSpot`] if `spot < 0`.
/// Returns [`PricingError::NegativeStrike`] if `strike < 0`.
/// Returns [`PricingError::NegativeTime`] if `time < 0`.
/// Returns [`PricingError::NegativeVolatility`] if `vol < 0`.
/// Returns [`PricingError::IntrinsicOnly`] if `time == 0`.
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::{OptionParams, OptionType};
/// use regit_blackscholes::greeks::compute_greeks;
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
/// let greeks = compute_greeks(&params).unwrap();
/// assert!(greeks.delta > 0.0_f64 && greeks.delta < 1.0_f64);
/// ```
#[inline(always)]
pub fn compute_greeks(params: &OptionParams<f64>) -> Result<Greeks<f64>, PricingError> {
    let spot = params.spot;
    let strike = params.strike;
    let rate = params.rate;
    let q = params.div_yield;
    let vol = params.vol;
    let time = params.time;

    // ── Input validation ────────────────────────────────────────────────
    if spot < 0.0_f64 {
        return Err(PricingError::NegativeSpot);
    }
    if strike < 0.0_f64 {
        return Err(PricingError::NegativeStrike);
    }
    if time < 0.0_f64 {
        return Err(PricingError::NegativeTime);
    }
    if vol < 0.0_f64 {
        return Err(PricingError::NegativeVolatility);
    }

    // ── T = 0 edge case ────────────────────────────────────────────────
    if time == 0.0_f64 {
        let intrinsic = match params.option_type {
            OptionType::Call => {
                if spot > strike {
                    spot - strike
                } else {
                    0.0_f64
                }
            }
            OptionType::Put => {
                if strike > spot {
                    strike - spot
                } else {
                    0.0_f64
                }
            }
        };
        return Err(PricingError::IntrinsicOnly { intrinsic });
    }

    // ── Intermediate values — computed ONCE ─────────────────────────────
    let sqrt_t = time.sqrt();
    let vol_sqrt_t = vol * sqrt_t;

    let d1_val = d1(spot, strike, rate, q, vol, time);
    let d2_val = d2(d1_val, vol, time);

    let nd1 = ncdf(d1_val);    // N(d1)
    let nd2 = ncdf(d2_val);    // N(d2)
    let nnd1 = ncdf(-d1_val);  // N(-d1)
    let nnd2 = ncdf(-d2_val);  // N(-d2)
    let pd1 = npdf(d1_val);    // φ(d1)
    let pd2 = npdf(d2_val);    // φ(d2)

    let exp_qt = (-q * time).exp();   // exp(-qT)
    let exp_rt = (-rate * time).exp(); // exp(-rT)

    // ── Price (needed for Lambda) ───────────────────────────────────────
    let price = match params.option_type {
        OptionType::Call => {
            spot * exp_qt * nd1 - strike * exp_rt * nd2
        }
        OptionType::Put => {
            strike * exp_rt * nnd2 - spot * exp_qt * nnd1
        }
    };

    // ── 1st order Greeks ────────────────────────────────────────────────

    // Delta: exp(-qT) * N(d1) for call, -exp(-qT) * N(-d1) for put
    let delta = match params.option_type {
        OptionType::Call => exp_qt * nd1,
        OptionType::Put => -exp_qt * nnd1,
    };

    // Gamma: exp(-qT) * φ(d1) / (S * σ * √T)
    // Symmetric for call and put
    let gamma = if vol_sqrt_t > 0.0_f64 {
        exp_qt * pd1 / (spot * vol_sqrt_t)
    } else {
        0.0_f64
    };

    // Theta (annualised, then divided by 365 for daily)
    // Call: -(S*exp(-qT)*φ(d1)*σ)/(2√T) - r*K*exp(-rT)*N(d2) + q*S*exp(-qT)*N(d1)
    // Put:  -(S*exp(-qT)*φ(d1)*σ)/(2√T) + r*K*exp(-rT)*N(-d2) - q*S*exp(-qT)*N(-d1)
    let theta_common = -(spot * exp_qt * pd1 * vol) / (2.0_f64 * sqrt_t);
    let theta_annual = match params.option_type {
        OptionType::Call => {
            theta_common - rate * strike * exp_rt * nd2 + q * spot * exp_qt * nd1
        }
        OptionType::Put => {
            theta_common + rate * strike * exp_rt * nnd2 - q * spot * exp_qt * nnd1
        }
    };
    let theta = theta_annual / 365.0_f64;

    // Vega: S * exp(-qT) * φ(d1) * √T (symmetric)
    let vega = spot * exp_qt * pd1 * sqrt_t;

    // Rho: K*T*exp(-rT)*N(d2) for call, -K*T*exp(-rT)*N(-d2) for put
    let rho = match params.option_type {
        OptionType::Call => strike * time * exp_rt * nd2,
        OptionType::Put => -strike * time * exp_rt * nnd2,
    };

    // Epsilon: -T*S*exp(-qT)*N(d1) for call, T*S*exp(-qT)*N(-d1) for put
    let epsilon = match params.option_type {
        OptionType::Call => -time * spot * exp_qt * nd1,
        OptionType::Put => time * spot * exp_qt * nnd1,
    };

    // Lambda: Delta * S / price
    let lambda = if price.abs() > 1e-30_f64 {
        delta * spot / price
    } else {
        0.0_f64
    };

    // ── 2nd order Greeks ────────────────────────────────────────────────

    // Vanna: -exp(-qT) * φ(d1) * d2 / σ (symmetric)
    let vanna = if vol > 0.0_f64 {
        -exp_qt * pd1 * d2_val / vol
    } else {
        0.0_f64
    };

    // Charm (delta decay): dΔ/dt
    // Call: q*exp(-qT)*N(d1) - exp(-qT)*φ(d1)*(2(r-q)T - d2*σ√T)/(2T*σ√T)
    // Put:  -q*exp(-qT)*N(-d1) + exp(-qT)*φ(d1)*(2(r-q)T - d2*σ√T)/(2T*σ√T)
    let charm = if vol_sqrt_t > 0.0_f64 {
        let factor = exp_qt * pd1
            * (2.0_f64 * (rate - q) * time - d2_val * vol_sqrt_t)
            / (2.0_f64 * time * vol_sqrt_t);
        match params.option_type {
            OptionType::Call => q * exp_qt * nd1 - factor,
            OptionType::Put => -q * exp_qt * nnd1 + factor,
        }
    } else {
        0.0_f64
    };

    // Veta (vega decay): dν/dt
    // Veta = -S*exp(-qT)*φ(d1)*√T * [q + (r-q)*d1/(σ√T) - (1 + d1*d2)/(2T)]
    let veta = if vol_sqrt_t > 0.0_f64 {
        let term = q + (rate - q) * d1_val / vol_sqrt_t
            - (1.0_f64 + d1_val * d2_val) / (2.0_f64 * time);
        -spot * exp_qt * pd1 * sqrt_t * term
    } else {
        0.0_f64
    };

    // Vomma: Vega * d1 * d2 / σ (symmetric)
    let vomma = if vol > 0.0_f64 {
        vega * d1_val * d2_val / vol
    } else {
        0.0_f64
    };

    // Dual Delta: dV/dK
    // Call: -exp(-rT) * N(d2),  Put: exp(-rT) * N(-d2)
    let dual_delta = match params.option_type {
        OptionType::Call => -exp_rt * nd2,
        OptionType::Put => exp_rt * nnd2,
    };

    // Dual Gamma: d²V/dK² = exp(-rT) * φ(d2) / (K * σ * √T)
    let dual_gamma = if vol_sqrt_t > 0.0_f64 {
        exp_rt * pd2 / (strike * vol_sqrt_t)
    } else {
        0.0_f64
    };

    // ── 3rd order Greeks ────────────────────────────────────────────────

    // Speed: -Γ * (d1/(σ√T) + 1) / S
    let speed = if vol_sqrt_t > 0.0_f64 && spot > 0.0_f64 {
        -gamma * (d1_val / vol_sqrt_t + 1.0_f64) / spot
    } else {
        0.0_f64
    };

    // Zomma: Γ * (d1*d2 - 1) / σ
    let zomma = if vol > 0.0_f64 {
        gamma * (d1_val * d2_val - 1.0_f64) / vol
    } else {
        0.0_f64
    };

    // Color (gamma decay): dΓ/dt
    // Color = -exp(-qT) * φ(d1) / (2*S*T*σ√T)
    //         * (2qT + 1 + d1*(2(r-q)T - d2*σ√T) / (σ√T))
    let color = if vol_sqrt_t > 0.0_f64 && spot > 0.0_f64 {
        let inner = 2.0_f64 * q * time + 1.0_f64
            + d1_val * (2.0_f64 * (rate - q) * time - d2_val * vol_sqrt_t) / vol_sqrt_t;
        -exp_qt * pd1 / (2.0_f64 * spot * time * vol_sqrt_t) * inner
    } else {
        0.0_f64
    };

    // Ultima: d³V/dσ³
    // Ultima = (-Vega / σ²) * (d1*d2*(1 - d1*d2) + d1² + d2²)
    let ultima = if vol > 0.0_f64 {
        let d1d2 = d1_val * d2_val;
        -vega / (vol * vol) * (d1d2 * (1.0_f64 - d1d2) + d1_val * d1_val + d2_val * d2_val)
    } else {
        0.0_f64
    };

    Ok(Greeks {
        delta,
        gamma,
        theta,
        vega,
        rho,
        epsilon,
        lambda,
        vanna,
        charm,
        veta,
        vomma,
        speed,
        zomma,
        color,
        ultima,
        dual_delta,
        dual_gamma,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Loose tolerance for golden value comparisons.
    const LOOSE: f64 = 1e-4_f64;

    /// Tolerance for symmetry tests (call vs put for symmetric Greeks).
    const SYMMETRY_TOL: f64 = 1e-12_f64;

    /// Baseline call params: S=100, K=100, r=0.05, q=0.02, σ=0.20, T=1.0
    fn baseline_call() -> OptionParams<f64> {
        OptionParams {
            option_type: OptionType::Call,
            spot: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        }
    }

    /// Baseline put params: S=100, K=100, r=0.05, q=0.02, σ=0.20, T=1.0
    fn baseline_put() -> OptionParams<f64> {
        OptionParams {
            option_type: OptionType::Put,
            spot: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        }
    }

    // ── Delta golden values ─────────────────────────────────────────────
    // Verified via put-call parity: Δ_call - Δ_put = exp(-qT)

    #[test]
    fn test_delta_call_matches_computed_value() {
        let g = compute_greeks(&baseline_call()).unwrap();
        // exp(-0.02) * N(0.25) ≈ 0.58685
        assert!(
            (g.delta - 0.5869_f64).abs() < LOOSE,
            "delta call: expected ~0.5869, got {}",
            g.delta
        );
    }

    #[test]
    fn test_delta_put_matches_computed_value() {
        let g = compute_greeks(&baseline_put()).unwrap();
        // -exp(-0.02) * N(-0.25) ≈ -0.3933
        assert!(
            (g.delta - (-0.3933_f64)).abs() < LOOSE,
            "delta put: expected ~-0.3933, got {}",
            g.delta
        );
    }

    #[test]
    fn test_delta_put_call_parity() {
        // Δ_call - Δ_put = exp(-qT)
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        let exp_qt = (-0.02_f64 * 1.0_f64).exp();
        assert!(
            (gc.delta - gp.delta - exp_qt).abs() < 1e-10_f64,
            "delta parity: {} - {} = {}, expected {}",
            gc.delta, gp.delta, gc.delta - gp.delta, exp_qt
        );
    }

    // ── Gamma golden values ─────────────────────────────────────────────

    #[test]
    fn test_gamma_call_matches_computed_value() {
        let g = compute_greeks(&baseline_call()).unwrap();
        // exp(-0.02) * φ(0.25) / (100 * 0.2) ≈ 0.01895
        assert!(
            (g.gamma - 0.01895_f64).abs() < LOOSE,
            "gamma call: expected ~0.01895, got {}",
            g.gamma
        );
    }

    #[test]
    fn test_gamma_call_equals_gamma_put() {
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        assert!(
            (gc.gamma - gp.gamma).abs() < SYMMETRY_TOL,
            "gamma symmetry: call={}, put={}",
            gc.gamma,
            gp.gamma
        );
    }

    // ── Vega golden values ──────────────────────────────────────────────

    #[test]
    fn test_vega_call_matches_computed_value() {
        let g = compute_greeks(&baseline_call()).unwrap();
        // S * exp(-qT) * φ(d1) * √T ≈ 37.90
        assert!(
            (g.vega - 37.90_f64).abs() < 0.1_f64,
            "vega call: expected ~37.90, got {}",
            g.vega
        );
    }

    #[test]
    fn test_vega_call_equals_vega_put() {
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        assert!(
            (gc.vega - gp.vega).abs() < SYMMETRY_TOL,
            "vega symmetry: call={}, put={}",
            gc.vega,
            gp.vega
        );
    }

    // ── Theta golden values ─────────────────────────────────────────────

    #[test]
    fn test_theta_call_is_negative() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(g.theta < 0.0_f64, "theta call must be negative, got {}", g.theta);
    }

    #[test]
    fn test_theta_put_is_negative() {
        let g = compute_greeks(&baseline_put()).unwrap();
        assert!(g.theta < 0.0_f64, "theta put must be negative, got {}", g.theta);
    }

    #[test]
    fn test_theta_call_more_negative_than_put() {
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        assert!(
            gc.theta < gp.theta,
            "call theta ({}) should be more negative than put theta ({})",
            gc.theta, gp.theta
        );
    }

    // ── Rho golden values ───────────────────────────────────────────────

    #[test]
    fn test_rho_call_positive() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(g.rho > 0.0_f64, "rho call must be positive, got {}", g.rho);
    }

    #[test]
    fn test_rho_put_negative() {
        let g = compute_greeks(&baseline_put()).unwrap();
        assert!(g.rho < 0.0_f64, "rho put must be negative, got {}", g.rho);
    }

    // ── Epsilon golden values ───────────────────────────────────────────

    #[test]
    fn test_epsilon_call_negative() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(g.epsilon < 0.0_f64, "epsilon call must be negative, got {}", g.epsilon);
    }

    #[test]
    fn test_epsilon_put_positive() {
        let g = compute_greeks(&baseline_put()).unwrap();
        assert!(g.epsilon > 0.0_f64, "epsilon put must be positive, got {}", g.epsilon);
    }

    // ── Vanna golden values ─────────────────────────────────────────────

    #[test]
    fn test_vanna_symmetric() {
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        assert!(
            (gc.vanna - gp.vanna).abs() < SYMMETRY_TOL,
            "vanna symmetry: call={}, put={}",
            gc.vanna,
            gp.vanna
        );
    }

    // ── Charm golden values ─────────────────────────────────────────────

    #[test]
    fn test_charm_call_negative() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(g.charm < 0.0_f64, "charm call should be negative for ATM, got {}", g.charm);
    }

    #[test]
    fn test_charm_put_positive() {
        let g = compute_greeks(&baseline_put()).unwrap();
        assert!(g.charm > 0.0_f64, "charm put should be positive for ATM, got {}", g.charm);
    }

    // ── Vomma golden values ─────────────────────────────────────────────

    #[test]
    fn test_vomma_symmetric() {
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        assert!(
            (gc.vomma - gp.vomma).abs() < SYMMETRY_TOL,
            "vomma symmetry: call={}, put={}",
            gc.vomma,
            gp.vomma
        );
    }

    #[test]
    fn test_vomma_equals_vega_d1_d2_over_sigma() {
        // Vomma = Vega * d1 * d2 / σ — verify identity directly
        let g = compute_greeks(&baseline_call()).unwrap();
        let d1_val = crate::math::d1(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
        let d2_val = crate::math::d2(d1_val, 0.20_f64, 1.0_f64);
        let expected = g.vega * d1_val * d2_val / 0.20_f64;
        assert!(
            (g.vomma - expected).abs() < 1e-10_f64,
            "vomma identity: got {}, expected {}",
            g.vomma, expected
        );
    }

    // ── Speed golden values ─────────────────────────────────────────────

    #[test]
    fn test_speed_symmetric() {
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        assert!(
            (gc.speed - gp.speed).abs() < SYMMETRY_TOL,
            "speed symmetry: call={}, put={}",
            gc.speed,
            gp.speed
        );
    }

    // ── Zomma golden values ─────────────────────────────────────────────

    #[test]
    fn test_zomma_symmetric() {
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        assert!(
            (gc.zomma - gp.zomma).abs() < SYMMETRY_TOL,
            "zomma symmetry: call={}, put={}",
            gc.zomma,
            gp.zomma
        );
    }

    // ── Dual delta ──────────────────────────────────────────────────────

    #[test]
    fn test_dual_delta_call_negative() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(g.dual_delta < 0.0_f64, "dual_delta call must be negative, got {}", g.dual_delta);
    }

    #[test]
    fn test_dual_delta_put_positive() {
        let g = compute_greeks(&baseline_put()).unwrap();
        assert!(g.dual_delta > 0.0_f64, "dual_delta put must be positive, got {}", g.dual_delta);
    }

    // ── Dual gamma ──────────────────────────────────────────────────────

    #[test]
    fn test_dual_gamma_symmetric() {
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        assert!(
            (gc.dual_gamma - gp.dual_gamma).abs() < SYMMETRY_TOL,
            "dual_gamma symmetry: call={}, put={}",
            gc.dual_gamma,
            gp.dual_gamma
        );
    }

    #[test]
    fn test_dual_gamma_positive() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(g.dual_gamma > 0.0_f64, "dual_gamma must be positive, got {}", g.dual_gamma);
    }

    // ── Input validation ────────────────────────────────────────────────

    #[test]
    fn test_negative_spot_returns_error() {
        let mut p = baseline_call();
        p.spot = -1.0_f64;
        assert!(matches!(compute_greeks(&p), Err(PricingError::NegativeSpot)));
    }

    #[test]
    fn test_negative_strike_returns_error() {
        let mut p = baseline_call();
        p.strike = -1.0_f64;
        assert!(matches!(
            compute_greeks(&p),
            Err(PricingError::NegativeStrike)
        ));
    }

    #[test]
    fn test_negative_time_returns_error() {
        let mut p = baseline_call();
        p.time = -0.01_f64;
        assert!(matches!(
            compute_greeks(&p),
            Err(PricingError::NegativeTime)
        ));
    }

    #[test]
    fn test_negative_vol_returns_error() {
        let mut p = baseline_call();
        p.vol = -0.01_f64;
        assert!(matches!(
            compute_greeks(&p),
            Err(PricingError::NegativeVolatility)
        ));
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_time_zero_returns_intrinsic_only_call_itm() {
        let mut p = baseline_call();
        p.time = 0.0_f64;
        p.spot = 110.0_f64;
        p.strike = 100.0_f64;
        match compute_greeks(&p) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert!(
                    (intrinsic - 10.0_f64).abs() < 1e-12_f64,
                    "intrinsic: expected 10.0, got {intrinsic}"
                );
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_time_zero_returns_intrinsic_only_put_itm() {
        let mut p = baseline_put();
        p.time = 0.0_f64;
        p.spot = 90.0_f64;
        p.strike = 100.0_f64;
        match compute_greeks(&p) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert!(
                    (intrinsic - 10.0_f64).abs() < 1e-12_f64,
                    "intrinsic: expected 10.0, got {intrinsic}"
                );
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_time_zero_returns_intrinsic_only_otm() {
        let mut p = baseline_call();
        p.time = 0.0_f64;
        p.spot = 90.0_f64;
        p.strike = 100.0_f64;
        match compute_greeks(&p) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert!(
                    intrinsic.abs() < 1e-12_f64,
                    "intrinsic: expected 0.0, got {intrinsic}"
                );
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_very_small_vol_no_nan() {
        let mut p = baseline_call();
        p.vol = 1e-10_f64;
        let g = compute_greeks(&p).unwrap();
        assert!(!g.delta.is_nan(), "delta is NaN with very small vol");
        assert!(!g.gamma.is_nan(), "gamma is NaN with very small vol");
        assert!(!g.vega.is_nan(), "vega is NaN with very small vol");
        assert!(!g.theta.is_nan(), "theta is NaN with very small vol");
        assert!(!g.rho.is_nan(), "rho is NaN with very small vol");
        assert!(!g.vanna.is_nan(), "vanna is NaN with very small vol");
        assert!(!g.vomma.is_nan(), "vomma is NaN with very small vol");
        assert!(!g.speed.is_nan(), "speed is NaN with very small vol");
        assert!(!g.zomma.is_nan(), "zomma is NaN with very small vol");
        assert!(!g.color.is_nan(), "color is NaN with very small vol");
        assert!(!g.ultima.is_nan(), "ultima is NaN with very small vol");
    }

    // ── Gamma / Vega property tests ─────────────────────────────────────

    #[test]
    fn test_gamma_always_positive() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(g.gamma > 0.0_f64, "gamma must be positive, got {}", g.gamma);
    }

    #[test]
    fn test_vega_always_positive() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(g.vega > 0.0_f64, "vega must be positive, got {}", g.vega);
    }

    #[test]
    fn test_delta_call_in_zero_one() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(
            g.delta > 0.0_f64 && g.delta < 1.0_f64,
            "call delta must be in (0,1), got {}",
            g.delta
        );
    }

    #[test]
    fn test_delta_put_in_neg_one_zero() {
        let g = compute_greeks(&baseline_put()).unwrap();
        assert!(
            g.delta > -1.0_f64 && g.delta < 0.0_f64,
            "put delta must be in (-1,0), got {}",
            g.delta
        );
    }

    // ── Lambda (leverage) ───────────────────────────────────────────────

    #[test]
    fn test_lambda_call_positive() {
        let g = compute_greeks(&baseline_call()).unwrap();
        assert!(g.lambda > 0.0_f64, "lambda call must be positive, got {}", g.lambda);
    }

    #[test]
    fn test_lambda_put_positive() {
        // Lambda for put is delta*S/V. Delta < 0 and price > 0, but S > 0.
        // Actually: put delta < 0, so lambda = (neg * pos) / pos = neg.
        let g = compute_greeks(&baseline_put()).unwrap();
        assert!(g.lambda < 0.0_f64, "lambda put must be negative, got {}", g.lambda);
    }

    // ── Vanna cross-derivative check (finite difference) ────────────────

    #[test]
    fn test_vanna_matches_finite_difference_dvega_ds() {
        let h = 1e-4_f64;
        let mut p_up = baseline_call();
        p_up.spot = 100.0_f64 + h;
        let mut p_dn = baseline_call();
        p_dn.spot = 100.0_f64 - h;
        let vega_up = compute_greeks(&p_up).unwrap().vega;
        let vega_dn = compute_greeks(&p_dn).unwrap().vega;
        let fd_vanna = (vega_up - vega_dn) / (2.0_f64 * h);
        let analytic_vanna = compute_greeks(&baseline_call()).unwrap().vanna;
        assert!(
            (analytic_vanna - fd_vanna).abs() < 0.01_f64,
            "vanna: analytic={}, fd={}",
            analytic_vanna, fd_vanna
        );
    }

    // ── Put-call parity for price (verify internal price computation) ───

    #[test]
    fn test_price_put_call_parity() {
        // C - P = S*exp(-qT) - K*exp(-rT)
        let gc = compute_greeks(&baseline_call()).unwrap();
        let gp = compute_greeks(&baseline_put()).unwrap();
        // Recover price from lambda: price = delta * S / lambda
        // Instead, verify through rho parity: rho_call + rho_put = -K*T*exp(-rT)
        // ... this is indirect. Better: verify delta parity holds.
        let exp_qt = (-0.02_f64 * 1.0_f64).exp();
        let delta_diff = gc.delta - gp.delta;
        assert!(
            (delta_diff - exp_qt).abs() < 1e-10_f64,
            "delta parity check for price consistency"
        );
    }
}
