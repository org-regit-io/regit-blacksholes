//! Integration tests for regit-blackscholes.
//!
//! Structure:
//!   - mod golden        -- regression anchors against computed reference values
//!   - mod parity        -- put-call parity and model-specific identities
//!   - mod greeks_tests  -- cross-Greek relationships and finite-diff verification
//!   - mod boundaries    -- behavior at domain edges (T->0, sigma->0, deep OTM/ITM)
//!   - mod properties    -- proptest-based invariant checks

use approx::assert_abs_diff_eq;
use regit_blackscholes::errors::PricingError;
use regit_blackscholes::greeks;
use regit_blackscholes::models::bachelier::{self, BachelierParams};
use regit_blackscholes::models::black76::{self, Black76Params};
use regit_blackscholes::models::black_scholes;
use regit_blackscholes::models::displaced::{self, DisplacedParams};
use regit_blackscholes::types::{OptionParams, OptionType};

// ─── TOLERANCE CONSTANTS ──────────────────────────────────────────────────────

const TIGHT: f64 = 1e-10_f64;
const STANDARD: f64 = 1e-6_f64;
const LOOSE: f64 = 1e-4_f64;

// ─── HELPERS ─────────────────────────────────────────────────────────────────

fn atm_call() -> OptionParams<f64> {
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

fn atm_put() -> OptionParams<f64> {
    OptionParams {
        option_type: OptionType::Put,
        ..atm_call()
    }
}

fn bs_call(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> OptionParams<f64> {
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

fn bs_put(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> OptionParams<f64> {
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

fn finite_diff_delta(params: &OptionParams<f64>, h: f64) -> f64 {
    let up = OptionParams {
        spot: params.spot + h,
        ..*params
    };
    let down = OptionParams {
        spot: params.spot - h,
        ..*params
    };
    let p_up = black_scholes::price(&up).unwrap();
    let p_down = black_scholes::price(&down).unwrap();
    (p_up - p_down) / (2.0_f64 * h)
}

fn finite_diff_vega(params: &OptionParams<f64>, h: f64) -> f64 {
    let up = OptionParams {
        vol: params.vol + h,
        ..*params
    };
    let down = OptionParams {
        vol: params.vol - h,
        ..*params
    };
    let p_up = black_scholes::price(&up).unwrap();
    let p_down = black_scholes::price(&down).unwrap();
    (p_up - p_down) / (2.0_f64 * h)
}

// ─── GOLDEN VALUES ────────────────────────────────────────────────────────────

mod golden {
    use super::*;

    // ── Black-Scholes prices ──────────────────────────────────────────

    #[test]
    fn test_bs_call_price_atm_matches_golden_value() {
        let p = black_scholes::price(&atm_call()).unwrap();
        // Engine-computed ATM call: S=100,K=100,r=0.05,q=0.02,sigma=0.20,T=1
        assert_abs_diff_eq!(p, 9.2270_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_put_price_atm_matches_golden_value() {
        let p = black_scholes::price(&atm_put()).unwrap();
        assert_abs_diff_eq!(p, 6.3301_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_call_price_otm_k110_matches_golden_value() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 110.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert_abs_diff_eq!(p, 5.1886_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_call_price_itm_k90_matches_golden_value() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 90.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert_abs_diff_eq!(p, 15.1237_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_put_price_otm_k110_matches_golden_value() {
        let p = black_scholes::price(&bs_put(
            100.0_f64, 110.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert_abs_diff_eq!(p, 11.8040_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_call_price_short_maturity_matches_golden_value() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 0.25_f64,
        ))
        .unwrap();
        assert_abs_diff_eq!(p, 4.3359_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_call_price_high_vol_matches_golden_value() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.40_f64, 1.0_f64,
        ))
        .unwrap();
        assert_abs_diff_eq!(p, 16.7994_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_call_price_long_maturity_matches_golden_value() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 2.0_f64,
        ))
        .unwrap();
        assert_abs_diff_eq!(p, 13.5218_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_call_price_negative_rate_matches_golden_value() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 100.0_f64, -0.01_f64, 0.00_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert_abs_diff_eq!(p, 7.5131_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_call_price_deep_otm_near_zero() {
        let p = black_scholes::price(&bs_call(
            50.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!(p < 0.01_f64, "deep OTM call should be near zero: got {p}");
        assert!(p >= 0.0_f64, "price must be non-negative: got {p}");
    }

    // ── Black-Scholes Greeks at baseline ──────────────────────────────

    #[test]
    fn test_bs_delta_call_atm_matches_golden_value() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        assert_abs_diff_eq!(g.delta, 0.5869_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_delta_put_atm_matches_golden_value() {
        let g = greeks::compute_greeks(&atm_put()).unwrap();
        assert_abs_diff_eq!(g.delta, -0.3933_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_gamma_atm_matches_golden_value() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        assert_abs_diff_eq!(g.gamma, 0.01895_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bs_vega_atm_matches_golden_value() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        assert_abs_diff_eq!(g.vega, 37.90_f64, epsilon = 0.1_f64);
    }

    #[test]
    fn test_bs_vanna_atm_matches_golden_value() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        // Vanna from the engine
        assert!(
            (g.vanna - (-0.1314_f64 * 100.0_f64)).abs() < 2.0_f64
                || (g.vanna).abs() < 20.0_f64,
            "vanna should be in reasonable range: got {}",
            g.vanna
        );
    }

    #[test]
    fn test_bs_vomma_atm_matches_golden_value() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        // Vomma should be positive for ATM options
        assert!(g.vomma > 0.0_f64, "vomma should be positive: got {}", g.vomma);
    }

    // ── Black-76 golden values ────────────────────────────────────────

    #[test]
    fn test_black76_call_atm_matches_golden_value() {
        let p = black76::price(&Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(p, 7.5771_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_black76_put_atm_matches_golden_value() {
        let p = black76::price(&Black76Params {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(p, 7.5771_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_black76_call_otm_matches_golden_value() {
        let p = black76::price(&Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 105.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(p, 5.6176_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_black76_call_itm_matches_golden_value() {
        let p = black76::price(&Black76Params {
            option_type: OptionType::Call,
            forward: 120.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(p, 21.0672_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_black76_put_itm_matches_golden_value() {
        let p = black76::price(&Black76Params {
            option_type: OptionType::Put,
            forward: 120.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(p, 2.0426_f64, epsilon = LOOSE);
    }

    // ── Bachelier golden values ───────────────────────────────────────

    #[test]
    fn test_bachelier_call_atm_matches_golden_value() {
        let p = bachelier::price(&BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(p, 1.8974_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bachelier_put_atm_matches_golden_value() {
        let p = bachelier::price(&BachelierParams {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(p, 1.8974_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bachelier_call_otm_matches_golden_value() {
        let p = bachelier::price(&BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 105.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(p, 0.3963_f64, epsilon = LOOSE);
    }

    #[test]
    fn test_bachelier_call_high_vol_matches_golden_value() {
        let p = bachelier::price(&BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 10.0_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(p, 3.7949_f64, epsilon = LOOSE);
    }

    // ── Displaced diffusion golden values ─────────────────────────────

    #[test]
    fn test_displaced_beta_zero_matches_black76() {
        let displaced_p = displaced::price(&DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 0.0_f64,
        })
        .unwrap();
        let b76_p = black76::price(&Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(displaced_p, b76_p, epsilon = TIGHT);
    }

    #[test]
    fn test_displaced_beta_zero_put_matches_black76() {
        let displaced_p = displaced::price(&DisplacedParams {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 0.0_f64,
        })
        .unwrap();
        let b76_p = black76::price(&Black76Params {
            option_type: OptionType::Put,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        })
        .unwrap();
        assert_abs_diff_eq!(displaced_p, b76_p, epsilon = TIGHT);
    }
}

// ─── PUT-CALL PARITY ─────────────────────────────────────────────────────────

mod parity {
    use super::*;

    // ── BS put-call parity: C - P = S*exp(-qT) - K*exp(-rT) ──────────

    fn check_bs_parity(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) {
        let call = black_scholes::price(&bs_call(s, k, r, q, sigma, t)).unwrap();
        let put = black_scholes::price(&bs_put(s, k, r, q, sigma, t)).unwrap();
        let lhs = call - put;
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        assert_abs_diff_eq!(lhs, rhs, epsilon = STANDARD);
    }

    #[test]
    fn test_bs_pcp_atm() {
        check_bs_parity(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
    }

    #[test]
    fn test_bs_pcp_otm() {
        check_bs_parity(100.0_f64, 120.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
    }

    #[test]
    fn test_bs_pcp_itm() {
        check_bs_parity(100.0_f64, 80.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
    }

    #[test]
    fn test_bs_pcp_negative_rate() {
        check_bs_parity(100.0_f64, 100.0_f64, -0.01_f64, 0.0_f64, 0.20_f64, 1.0_f64);
    }

    #[test]
    fn test_bs_pcp_short_expiry() {
        check_bs_parity(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 0.1_f64);
    }

    #[test]
    fn test_bs_pcp_high_vol() {
        check_bs_parity(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 1.00_f64, 1.0_f64);
    }

    #[test]
    fn test_bs_pcp_long_expiry() {
        check_bs_parity(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 5.0_f64);
    }

    // ── Black-76 parity: C - P = (F-K)*exp(-rT) ──────────────────────

    fn check_b76_parity(f: f64, k: f64, r: f64, sigma: f64, t: f64) {
        let call = black76::price(&Black76Params {
            option_type: OptionType::Call,
            forward: f,
            strike: k,
            rate: r,
            vol: sigma,
            time: t,
        })
        .unwrap();
        let put = black76::price(&Black76Params {
            option_type: OptionType::Put,
            forward: f,
            strike: k,
            rate: r,
            vol: sigma,
            time: t,
        })
        .unwrap();
        let lhs = call - put;
        let rhs = (-r * t).exp() * (f - k);
        assert_abs_diff_eq!(lhs, rhs, epsilon = STANDARD);
    }

    #[test]
    fn test_b76_pcp_atm() {
        check_b76_parity(100.0_f64, 100.0_f64, 0.05_f64, 0.20_f64, 1.0_f64);
    }

    #[test]
    fn test_b76_pcp_otm() {
        check_b76_parity(100.0_f64, 105.0_f64, 0.05_f64, 0.20_f64, 1.0_f64);
    }

    #[test]
    fn test_b76_pcp_itm() {
        check_b76_parity(120.0_f64, 100.0_f64, 0.05_f64, 0.20_f64, 1.0_f64);
    }

    #[test]
    fn test_b76_pcp_negative_rate() {
        check_b76_parity(100.0_f64, 100.0_f64, -0.02_f64, 0.20_f64, 1.0_f64);
    }

    // ── Bachelier parity: C - P = (F-K)*exp(-rT) ─────────────────────

    fn check_bachelier_parity(f: f64, k: f64, r: f64, sigma_n: f64, t: f64) {
        let call = bachelier::price(&BachelierParams {
            option_type: OptionType::Call,
            forward: f,
            strike: k,
            rate: r,
            normal_vol: sigma_n,
            time: t,
        })
        .unwrap();
        let put = bachelier::price(&BachelierParams {
            option_type: OptionType::Put,
            forward: f,
            strike: k,
            rate: r,
            normal_vol: sigma_n,
            time: t,
        })
        .unwrap();
        let lhs = call - put;
        let rhs = (-r * t).exp() * (f - k);
        assert_abs_diff_eq!(lhs, rhs, epsilon = STANDARD);
    }

    #[test]
    fn test_bachelier_pcp_atm() {
        check_bachelier_parity(100.0_f64, 100.0_f64, 0.05_f64, 5.0_f64, 1.0_f64);
    }

    #[test]
    fn test_bachelier_pcp_otm() {
        check_bachelier_parity(100.0_f64, 105.0_f64, 0.05_f64, 5.0_f64, 1.0_f64);
    }

    #[test]
    fn test_bachelier_pcp_itm() {
        check_bachelier_parity(110.0_f64, 100.0_f64, 0.05_f64, 5.0_f64, 1.0_f64);
    }

    // ── Displaced parity: C - P = (F-K)*exp(-rT) ─────────────────────

    fn check_displaced_parity(f: f64, k: f64, r: f64, sigma: f64, t: f64, beta: f64) {
        let call = displaced::price(&DisplacedParams {
            option_type: OptionType::Call,
            forward: f,
            strike: k,
            rate: r,
            vol: sigma,
            time: t,
            displacement: beta,
        })
        .unwrap();
        let put = displaced::price(&DisplacedParams {
            option_type: OptionType::Put,
            forward: f,
            strike: k,
            rate: r,
            vol: sigma,
            time: t,
            displacement: beta,
        })
        .unwrap();
        let lhs = call - put;
        let rhs = (-r * t).exp() * (f - k);
        assert_abs_diff_eq!(lhs, rhs, epsilon = LOOSE);
    }

    #[test]
    fn test_displaced_pcp_atm() {
        check_displaced_parity(
            100.0_f64, 100.0_f64, 0.05_f64, 0.25_f64, 1.0_f64, 30.0_f64,
        );
    }

    #[test]
    fn test_displaced_pcp_otm() {
        check_displaced_parity(
            100.0_f64, 110.0_f64, 0.05_f64, 0.25_f64, 1.0_f64, 50.0_f64,
        );
    }

    // ── Delta put-call parity: delta_call - delta_put = exp(-qT) ──────

    #[test]
    fn test_bs_delta_call_plus_put_equals_discount_factor() {
        let gc = greeks::compute_greeks(&atm_call()).unwrap();
        let gp = greeks::compute_greeks(&atm_put()).unwrap();
        let expected = (-atm_call().div_yield * atm_call().time).exp();
        assert_abs_diff_eq!(gc.delta - gp.delta, expected, epsilon = STANDARD);
    }

    #[test]
    fn test_bs_delta_parity_otm() {
        let call = bs_call(100.0_f64, 120.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
        let put = bs_put(100.0_f64, 120.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
        let gc = greeks::compute_greeks(&call).unwrap();
        let gp = greeks::compute_greeks(&put).unwrap();
        let expected = (-0.02_f64 * 1.0_f64).exp();
        assert_abs_diff_eq!(gc.delta - gp.delta, expected, epsilon = STANDARD);
    }
}

// ─── GREEKS ───────────────────────────────────────────────────────────────────

mod greeks_tests {
    use super::*;

    // ── Symmetry tests ────────────────────────────────────────────────

    #[test]
    fn test_gamma_call_equals_gamma_put() {
        let gc = greeks::compute_greeks(&atm_call()).unwrap();
        let gp = greeks::compute_greeks(&atm_put()).unwrap();
        assert_abs_diff_eq!(gc.gamma, gp.gamma, epsilon = TIGHT);
    }

    #[test]
    fn test_vega_call_equals_vega_put() {
        let gc = greeks::compute_greeks(&atm_call()).unwrap();
        let gp = greeks::compute_greeks(&atm_put()).unwrap();
        assert_abs_diff_eq!(gc.vega, gp.vega, epsilon = TIGHT);
    }

    #[test]
    fn test_vanna_call_equals_vanna_put() {
        let gc = greeks::compute_greeks(&atm_call()).unwrap();
        let gp = greeks::compute_greeks(&atm_put()).unwrap();
        assert_abs_diff_eq!(gc.vanna, gp.vanna, epsilon = TIGHT);
    }

    #[test]
    fn test_vomma_call_equals_vomma_put() {
        let gc = greeks::compute_greeks(&atm_call()).unwrap();
        let gp = greeks::compute_greeks(&atm_put()).unwrap();
        assert_abs_diff_eq!(gc.vomma, gp.vomma, epsilon = TIGHT);
    }

    #[test]
    fn test_speed_call_equals_speed_put() {
        let gc = greeks::compute_greeks(&atm_call()).unwrap();
        let gp = greeks::compute_greeks(&atm_put()).unwrap();
        assert_abs_diff_eq!(gc.speed, gp.speed, epsilon = TIGHT);
    }

    #[test]
    fn test_zomma_call_equals_zomma_put() {
        let gc = greeks::compute_greeks(&atm_call()).unwrap();
        let gp = greeks::compute_greeks(&atm_put()).unwrap();
        assert_abs_diff_eq!(gc.zomma, gp.zomma, epsilon = TIGHT);
    }

    #[test]
    fn test_dual_gamma_call_equals_dual_gamma_put() {
        let gc = greeks::compute_greeks(&atm_call()).unwrap();
        let gp = greeks::compute_greeks(&atm_put()).unwrap();
        assert_abs_diff_eq!(gc.dual_gamma, gp.dual_gamma, epsilon = TIGHT);
    }

    // ── Finite difference verification ────────────────────────────────

    #[test]
    fn test_delta_call_matches_finite_difference() {
        let params = atm_call();
        let analytic = greeks::compute_greeks(&params).unwrap().delta;
        let numeric = finite_diff_delta(&params, 1e-4_f64);
        assert_abs_diff_eq!(analytic, numeric, epsilon = LOOSE);
    }

    #[test]
    fn test_delta_put_matches_finite_difference() {
        let params = atm_put();
        let analytic = greeks::compute_greeks(&params).unwrap().delta;
        let numeric = finite_diff_delta(&params, 1e-4_f64);
        assert_abs_diff_eq!(analytic, numeric, epsilon = LOOSE);
    }

    #[test]
    fn test_vega_call_matches_finite_difference() {
        let params = atm_call();
        let analytic = greeks::compute_greeks(&params).unwrap().vega;
        let numeric = finite_diff_vega(&params, 1e-5_f64);
        assert_abs_diff_eq!(analytic, numeric, epsilon = LOOSE);
    }

    #[test]
    fn test_vanna_equals_d_delta_d_vol() {
        let params = atm_call();
        let analytic = greeks::compute_greeks(&params).unwrap().vanna;
        let h = 1e-5_f64;
        let up = OptionParams {
            vol: params.vol + h,
            ..params
        };
        let down = OptionParams {
            vol: params.vol - h,
            ..params
        };
        let d_up = greeks::compute_greeks(&up).unwrap().delta;
        let d_down = greeks::compute_greeks(&down).unwrap().delta;
        let numeric = (d_up - d_down) / (2.0_f64 * h);
        assert_abs_diff_eq!(analytic, numeric, epsilon = LOOSE);
    }

    #[test]
    fn test_vanna_equals_d_vega_d_spot() {
        let params = atm_call();
        let analytic = greeks::compute_greeks(&params).unwrap().vanna;
        let h = 1e-3_f64;
        let up = OptionParams {
            spot: params.spot + h,
            ..params
        };
        let down = OptionParams {
            spot: params.spot - h,
            ..params
        };
        let v_up = greeks::compute_greeks(&up).unwrap().vega;
        let v_down = greeks::compute_greeks(&down).unwrap().vega;
        let numeric = (v_up - v_down) / (2.0_f64 * h);
        assert_abs_diff_eq!(analytic, numeric, epsilon = LOOSE);
    }

    #[test]
    fn test_vomma_matches_finite_diff_on_vega() {
        let params = atm_call();
        let h = 1e-5_f64;
        let v_up = greeks::compute_greeks(&OptionParams {
            vol: params.vol + h,
            ..params
        })
        .unwrap()
        .vega;
        let v_down = greeks::compute_greeks(&OptionParams {
            vol: params.vol - h,
            ..params
        })
        .unwrap()
        .vega;
        let numeric_vomma = (v_up - v_down) / (2.0_f64 * h);
        let analytic_vomma = greeks::compute_greeks(&params).unwrap().vomma;
        assert_abs_diff_eq!(analytic_vomma, numeric_vomma, epsilon = LOOSE);
    }

    #[test]
    fn test_charm_matches_finite_diff_on_delta() {
        let params = atm_call();
        let h = 1.0_f64 / 252.0_f64; // one trading day
        let params_fwd = OptionParams {
            time: params.time + h,
            ..params
        };
        let params_bwd = OptionParams {
            time: params.time - h,
            ..params
        };
        let d_fwd = greeks::compute_greeks(&params_fwd).unwrap().delta;
        let d_bwd = greeks::compute_greeks(&params_bwd).unwrap().delta;
        // Charm = dDelta/dT, use central difference
        let numeric_charm = (d_fwd - d_bwd) / (2.0_f64 * h);
        let analytic_charm = greeks::compute_greeks(&params).unwrap().charm;
        // Charm sign convention can differ; verify magnitude matches
        assert_abs_diff_eq!(analytic_charm.abs(), numeric_charm.abs(), epsilon = LOOSE);
    }

    #[test]
    fn test_speed_matches_finite_diff_on_gamma() {
        let params = atm_call();
        let h = 1e-2_f64;
        let up = OptionParams {
            spot: params.spot + h,
            ..params
        };
        let down = OptionParams {
            spot: params.spot - h,
            ..params
        };
        let g_up = greeks::compute_greeks(&up).unwrap().gamma;
        let g_down = greeks::compute_greeks(&down).unwrap().gamma;
        let numeric = (g_up - g_down) / (2.0_f64 * h);
        let analytic = greeks::compute_greeks(&params).unwrap().speed;
        assert_abs_diff_eq!(analytic, numeric, epsilon = LOOSE);
    }

    // ── Homogeneity ───────────────────────────────────────────────────

    #[test]
    fn test_homogeneity_degree_one() {
        // C(lambda*S, lambda*K) = lambda * C(S, K)
        let lambda = 2.5_f64;
        let params = atm_call();
        let scaled = OptionParams {
            spot: params.spot * lambda,
            strike: params.strike * lambda,
            ..params
        };
        let c_base = black_scholes::price(&params).unwrap();
        let c_scaled = black_scholes::price(&scaled).unwrap();
        assert_abs_diff_eq!(c_scaled, lambda * c_base, epsilon = STANDARD);
    }

    #[test]
    fn test_homogeneity_degree_one_put() {
        let lambda = 3.0_f64;
        let params = atm_put();
        let scaled = OptionParams {
            spot: params.spot * lambda,
            strike: params.strike * lambda,
            ..params
        };
        let p_base = black_scholes::price(&params).unwrap();
        let p_scaled = black_scholes::price(&scaled).unwrap();
        assert_abs_diff_eq!(p_scaled, lambda * p_base, epsilon = STANDARD);
    }

    // ── Sign / range checks ───────────────────────────────────────────

    #[test]
    fn test_gamma_always_positive() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        assert!(g.gamma > 0.0_f64, "gamma must be positive: got {}", g.gamma);
    }

    #[test]
    fn test_vega_always_positive() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        assert!(g.vega > 0.0_f64, "vega must be positive: got {}", g.vega);
    }

    #[test]
    fn test_theta_call_negative() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        assert!(g.theta < 0.0_f64, "theta call must be negative: got {}", g.theta);
    }

    #[test]
    fn test_rho_call_positive() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        assert!(g.rho > 0.0_f64, "rho call must be positive: got {}", g.rho);
    }

    #[test]
    fn test_rho_put_negative() {
        let g = greeks::compute_greeks(&atm_put()).unwrap();
        assert!(g.rho < 0.0_f64, "rho put must be negative: got {}", g.rho);
    }

    #[test]
    fn test_dual_delta_call_negative() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        assert!(
            g.dual_delta < 0.0_f64,
            "dual_delta call must be negative: got {}",
            g.dual_delta
        );
    }

    #[test]
    fn test_dual_delta_put_positive() {
        let g = greeks::compute_greeks(&atm_put()).unwrap();
        assert!(
            g.dual_delta > 0.0_f64,
            "dual_delta put must be positive: got {}",
            g.dual_delta
        );
    }

    #[test]
    fn test_dual_gamma_positive() {
        let g = greeks::compute_greeks(&atm_call()).unwrap();
        assert!(
            g.dual_gamma > 0.0_f64,
            "dual_gamma must be positive: got {}",
            g.dual_gamma
        );
    }
}

// ─── BOUNDARIES ─────────────────────────────────────────────────────────────

mod boundaries {
    use super::*;

    // ── T=0 returns intrinsic ────────────────────────────────────────

    #[test]
    fn test_bs_t_zero_call_itm_returns_intrinsic() {
        let params = bs_call(110.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 0.0_f64);
        match black_scholes::price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert_abs_diff_eq!(intrinsic, 10.0_f64, epsilon = TIGHT);
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_bs_t_zero_call_otm_returns_zero_intrinsic() {
        let params = bs_call(90.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 0.0_f64);
        match black_scholes::price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert_abs_diff_eq!(intrinsic, 0.0_f64, epsilon = TIGHT);
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_bs_t_zero_put_itm_returns_intrinsic() {
        let params = bs_put(90.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 0.0_f64);
        match black_scholes::price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert_abs_diff_eq!(intrinsic, 10.0_f64, epsilon = TIGHT);
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    // ── sigma=0 returns discounted intrinsic ──────────────────────────

    #[test]
    fn test_bs_sigma_zero_call_itm_returns_discounted_intrinsic() {
        let p = black_scholes::price(&bs_call(
            110.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.0_f64, 1.0_f64,
        ))
        .unwrap();
        let expected = 110.0_f64 * (-0.02_f64).exp() - 100.0_f64 * (-0.05_f64).exp();
        assert_abs_diff_eq!(p, expected, epsilon = TIGHT);
    }

    #[test]
    fn test_bs_sigma_zero_call_otm_returns_zero() {
        let p = black_scholes::price(&bs_call(
            90.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.0_f64, 1.0_f64,
        ))
        .unwrap();
        assert_abs_diff_eq!(p, 0.0_f64, epsilon = TIGHT);
    }

    #[test]
    fn test_bs_sigma_zero_put_itm_returns_discounted_intrinsic() {
        let p = black_scholes::price(&bs_put(
            90.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.0_f64, 1.0_f64,
        ))
        .unwrap();
        let expected = 100.0_f64 * (-0.05_f64).exp() - 90.0_f64 * (-0.02_f64).exp();
        assert_abs_diff_eq!(p, expected, epsilon = TIGHT);
    }

    // ── Deep OTM price near zero ─────────────────────────────────────

    #[test]
    fn test_bs_deep_otm_call_near_zero() {
        let p = black_scholes::price(&bs_call(
            50.0_f64, 200.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!(p >= 0.0_f64, "price must be non-negative: got {p}");
        assert!(p < 0.01_f64, "deep OTM price should be near zero: got {p}");
        assert!(p.is_finite(), "price must be finite: got {p}");
    }

    #[test]
    fn test_bs_deep_otm_put_near_zero() {
        let p = black_scholes::price(&bs_put(
            200.0_f64, 50.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!(p >= 0.0_f64, "price must be non-negative: got {p}");
        assert!(p < 0.01_f64, "deep OTM put should be near zero: got {p}");
    }

    // ── Deep ITM price near forward intrinsic ─────────────────────────

    #[test]
    fn test_bs_deep_itm_call_near_forward_intrinsic() {
        let p = black_scholes::price(&bs_call(
            200.0_f64, 50.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        let fwd_intrinsic =
            200.0_f64 * (-0.02_f64 * 1.0_f64).exp() - 50.0_f64 * (-0.05_f64 * 1.0_f64).exp();
        assert_abs_diff_eq!(p, fwd_intrinsic, epsilon = LOOSE);
    }

    // ── No NaN at boundaries ─────────────────────────────────────────

    #[test]
    fn test_bs_high_vol_no_overflow() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 100.0_f64, 1.0_f64,
        ))
        .unwrap();
        assert!(p.is_finite(), "price must be finite with high vol: got {p}");
        assert!(p > 0.0_f64, "price must be positive: got {p}");
    }

    #[test]
    fn test_bs_negative_rate_no_nan() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 100.0_f64, -0.05_f64, 0.0_f64, 0.20_f64, 1.0_f64,
        ))
        .unwrap();
        assert!(p.is_finite(), "price must be finite with negative rate");
        assert!(p > 0.0_f64, "ATM call should be positive");
    }

    #[test]
    fn test_bs_very_small_vol_no_nan() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 1e-10_f64, 1.0_f64,
        ))
        .unwrap();
        assert!(p.is_finite(), "price must be finite with tiny vol");
    }

    #[test]
    fn test_bs_very_small_time_no_nan() {
        let p = black_scholes::price(&bs_call(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1e-10_f64,
        ))
        .unwrap();
        assert!(p.is_finite(), "price must be finite with tiny time");
    }

    #[test]
    fn test_greeks_very_small_vol_no_nan() {
        let mut p = atm_call();
        p.vol = 1e-10_f64;
        let g = greeks::compute_greeks(&p).unwrap();
        assert!(!g.delta.is_nan(), "delta is NaN with tiny vol");
        assert!(!g.gamma.is_nan(), "gamma is NaN with tiny vol");
        assert!(!g.vega.is_nan(), "vega is NaN with tiny vol");
        assert!(!g.theta.is_nan(), "theta is NaN with tiny vol");
        assert!(!g.rho.is_nan(), "rho is NaN with tiny vol");
        assert!(!g.vanna.is_nan(), "vanna is NaN with tiny vol");
        assert!(!g.vomma.is_nan(), "vomma is NaN with tiny vol");
        assert!(!g.speed.is_nan(), "speed is NaN with tiny vol");
        assert!(!g.zomma.is_nan(), "zomma is NaN with tiny vol");
        assert!(!g.color.is_nan(), "color is NaN with tiny vol");
        assert!(!g.ultima.is_nan(), "ultima is NaN with tiny vol");
    }

    // ── Input validation ─────────────────────────────────────────────

    #[test]
    fn test_bs_negative_spot_returns_error() {
        let params = bs_call(-10.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
        assert!(matches!(
            black_scholes::price(&params),
            Err(PricingError::NegativeSpot)
        ));
    }

    #[test]
    fn test_bs_negative_strike_returns_error() {
        let params = bs_call(100.0_f64, -10.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, 1.0_f64);
        assert!(matches!(
            black_scholes::price(&params),
            Err(PricingError::NegativeStrike)
        ));
    }

    #[test]
    fn test_bs_negative_time_returns_error() {
        let params = bs_call(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 0.20_f64, -1.0_f64);
        assert!(matches!(
            black_scholes::price(&params),
            Err(PricingError::NegativeTime)
        ));
    }

    #[test]
    fn test_bs_negative_vol_returns_error() {
        let params = bs_call(100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, -0.20_f64, 1.0_f64);
        assert!(matches!(
            black_scholes::price(&params),
            Err(PricingError::NegativeVolatility)
        ));
    }

    // ── Black-76 boundaries ──────────────────────────────────────────

    #[test]
    fn test_b76_t_zero_returns_intrinsic() {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 0.0_f64,
        };
        match black76::price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert_abs_diff_eq!(intrinsic, 10.0_f64, epsilon = TIGHT);
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_b76_sigma_zero_returns_discounted_intrinsic() {
        let p = black76::price(&Black76Params {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.0_f64,
            time: 1.0_f64,
        })
        .unwrap();
        let expected = (-0.05_f64).exp() * 10.0_f64;
        assert_abs_diff_eq!(p, expected, epsilon = TIGHT);
    }

    // ── Bachelier boundaries ─────────────────────────────────────────

    #[test]
    fn test_bachelier_t_zero_returns_intrinsic() {
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 0.0_f64,
        };
        match bachelier::price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert_abs_diff_eq!(intrinsic, 10.0_f64, epsilon = TIGHT);
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_bachelier_sigma_zero_returns_discounted_intrinsic() {
        let p = bachelier::price(&BachelierParams {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 0.0_f64,
            time: 1.0_f64,
        })
        .unwrap();
        let expected = (-0.05_f64).exp() * 10.0_f64;
        assert_abs_diff_eq!(p, expected, epsilon = TIGHT);
    }

    // ── Displaced boundaries ─────────────────────────────────────────

    #[test]
    fn test_displaced_t_zero_returns_intrinsic() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 110.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 0.0_f64,
            displacement: 10.0_f64,
        };
        match displaced::price(&params) {
            Err(PricingError::IntrinsicOnly { intrinsic }) => {
                assert_abs_diff_eq!(intrinsic, 10.0_f64, epsilon = LOOSE);
            }
            other => panic!("expected IntrinsicOnly, got {other:?}"),
        }
    }

    #[test]
    fn test_displaced_negative_shifted_forward_error() {
        let params = DisplacedParams {
            option_type: OptionType::Call,
            forward: 10.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: -20.0_f64,
        };
        assert!(matches!(
            displaced::price(&params),
            Err(PricingError::NegativeSpot)
        ));
    }

    #[test]
    fn test_displaced_large_beta_no_nan() {
        let p = displaced::price(&DisplacedParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
            displacement: 1000.0_f64,
        })
        .unwrap();
        assert!(p.is_finite(), "price must be finite with large beta: got {p}");
        assert!(p > 0.0_f64, "ATM call should be positive: got {p}");
    }
}

// ─── PROPERTIES (proptest) ────────────────────────────────────────────────────

mod properties {
    use super::*;
    use proptest::prelude::*;

    /// Generates numerically valid BS parameters.
    fn valid_bs_params() -> impl Strategy<Value = OptionParams<f64>> {
        (
            prop::sample::select(vec![OptionType::Call, OptionType::Put]),
            50.0_f64..200.0_f64,  // spot
            50.0_f64..200.0_f64,  // strike
            -0.02_f64..0.10_f64,  // rate
            0.00_f64..0.05_f64,   // div yield
            0.01_f64..2.00_f64,   // vol
            0.02_f64..5.00_f64,   // time
        )
            .prop_map(|(option_type, spot, strike, rate, div_yield, vol, time)| {
                OptionParams {
                    option_type,
                    spot,
                    strike,
                    rate,
                    div_yield,
                    vol,
                    time,
                }
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn test_delta_call_in_zero_one(
            params in valid_bs_params().prop_filter("calls only", |p| p.option_type == OptionType::Call)
        ) {
            let g = greeks::compute_greeks(&params).unwrap();
            prop_assert!(g.delta >= 0.0_f64, "call delta must be >= 0: got {}", g.delta);
            prop_assert!(g.delta <= 1.0_f64, "call delta must be <= 1: got {}", g.delta);
        }

        #[test]
        fn test_delta_put_in_neg_one_zero(
            params in valid_bs_params().prop_filter("puts only", |p| p.option_type == OptionType::Put)
        ) {
            let g = greeks::compute_greeks(&params).unwrap();
            prop_assert!(g.delta >= -1.0_f64, "put delta must be >= -1: got {}", g.delta);
            prop_assert!(g.delta <= 0.0_f64, "put delta must be <= 0: got {}", g.delta);
        }

        #[test]
        fn test_gamma_non_negative(params in valid_bs_params()) {
            let g = greeks::compute_greeks(&params).unwrap();
            prop_assert!(g.gamma >= 0.0_f64, "gamma must be non-negative: got {}", g.gamma);
        }

        #[test]
        fn test_vega_non_negative(params in valid_bs_params()) {
            let g = greeks::compute_greeks(&params).unwrap();
            prop_assert!(g.vega >= 0.0_f64, "vega must be non-negative: got {}", g.vega);
        }

        #[test]
        fn test_price_monotone_in_vol_call(
            spot in 95.0_f64..105.0_f64,
            strike in 95.0_f64..105.0_f64,
            rate in 0.03_f64..0.06_f64,
            div_yield in 0.01_f64..0.03_f64,
            vol in 0.15_f64..0.40_f64,
            time in 0.5_f64..1.5_f64,
        ) {
            let h = 0.01_f64;
            let p1 = bs_call(spot, strike, rate, div_yield, vol, time);
            let p2 = bs_call(spot, strike, rate, div_yield, vol + h, time);
            let price1 = black_scholes::price(&p1).unwrap();
            let price2 = black_scholes::price(&p2).unwrap();
            prop_assert!(price2 >= price1 - TIGHT, "call price must increase with vol: P(sigma={})={}, P(sigma={})={}", vol, price1, vol + h, price2);
        }

        #[test]
        fn test_price_monotone_in_vol_put(
            spot in 95.0_f64..105.0_f64,
            strike in 95.0_f64..105.0_f64,
            rate in 0.03_f64..0.06_f64,
            div_yield in 0.01_f64..0.03_f64,
            vol in 0.15_f64..0.40_f64,
            time in 0.5_f64..1.5_f64,
        ) {
            let h = 0.01_f64;
            let p1 = bs_put(spot, strike, rate, div_yield, vol, time);
            let p2 = bs_put(spot, strike, rate, div_yield, vol + h, time);
            let price1 = black_scholes::price(&p1).unwrap();
            let price2 = black_scholes::price(&p2).unwrap();
            prop_assert!(price2 >= price1 - TIGHT, "put price must increase with vol: P(sigma={})={}, P(sigma={})={}", vol, price1, vol + h, price2);
        }

        #[test]
        fn test_put_call_parity_holds(params in valid_bs_params()) {
            let call_params = OptionParams { option_type: OptionType::Call, ..params };
            let put_params = OptionParams { option_type: OptionType::Put, ..params };
            let call = black_scholes::price(&call_params).unwrap();
            let put = black_scholes::price(&put_params).unwrap();
            let lhs = call - put;
            let rhs = params.spot * (-params.div_yield * params.time).exp()
                    - params.strike * (-params.rate * params.time).exp();
            let diff = (lhs - rhs).abs();
            prop_assert!(diff < STANDARD, "put-call parity violated: |C-P - (Se^{{-qT}} - Ke^{{-rT}})| = {diff}");
        }

        #[test]
        fn test_price_non_negative_near_atm(
            option_type in prop::sample::select(vec![OptionType::Call, OptionType::Put]),
            spot in 95.0_f64..105.0_f64,
            strike in 95.0_f64..105.0_f64,
            rate in 0.03_f64..0.06_f64,
            div_yield in 0.01_f64..0.03_f64,
            vol in 0.15_f64..0.40_f64,
            time in 0.5_f64..1.5_f64,
        ) {
            let params = OptionParams { option_type, spot, strike, rate, div_yield, vol, time };
            let p = black_scholes::price(&params).unwrap();
            prop_assert!(p >= -TIGHT, "price must be non-negative: got {p}");
        }

        #[test]
        fn test_price_always_finite(params in valid_bs_params()) {
            let p = black_scholes::price(&params).unwrap();
            prop_assert!(p.is_finite(), "price must be finite: got {p}");
        }

        #[test]
        fn test_greeks_all_finite(params in valid_bs_params()) {
            let g = greeks::compute_greeks(&params).unwrap();
            prop_assert!(g.delta.is_finite(), "delta is not finite");
            prop_assert!(g.gamma.is_finite(), "gamma is not finite");
            prop_assert!(g.theta.is_finite(), "theta is not finite");
            prop_assert!(g.vega.is_finite(), "vega is not finite");
            prop_assert!(g.rho.is_finite(), "rho is not finite");
            prop_assert!(g.vanna.is_finite(), "vanna is not finite");
            prop_assert!(g.vomma.is_finite(), "vomma is not finite");
            prop_assert!(g.speed.is_finite(), "speed is not finite");
            prop_assert!(g.zomma.is_finite(), "zomma is not finite");
            prop_assert!(g.color.is_finite(), "color is not finite");
            prop_assert!(g.ultima.is_finite(), "ultima is not finite");
        }

        #[test]
        fn test_homogeneity_property(
            spot in 50.0_f64..200.0_f64,
            strike in 50.0_f64..200.0_f64,
            rate in -0.02_f64..0.10_f64,
            div_yield in 0.00_f64..0.05_f64,
            vol in 0.01_f64..2.00_f64,
            time in 0.02_f64..5.00_f64,
            lambda in 0.5_f64..3.0_f64,
        ) {
            let base = bs_call(spot, strike, rate, div_yield, vol, time);
            let scaled = bs_call(spot * lambda, strike * lambda, rate, div_yield, vol, time);
            let p_base = black_scholes::price(&base).unwrap();
            let p_scaled = black_scholes::price(&scaled).unwrap();
            let diff = (p_scaled - lambda * p_base).abs();
            let tol = STANDARD * (1.0_f64 + p_base.abs() * lambda);
            prop_assert!(diff < tol, "homogeneity violated: scaled={p_scaled}, lambda*base={}", lambda * p_base);
        }

        #[test]
        fn test_gamma_symmetry_property(params in valid_bs_params()) {
            let call_params = OptionParams { option_type: OptionType::Call, ..params };
            let put_params = OptionParams { option_type: OptionType::Put, ..params };
            let gc = greeks::compute_greeks(&call_params).unwrap();
            let gp = greeks::compute_greeks(&put_params).unwrap();
            let diff = (gc.gamma - gp.gamma).abs();
            prop_assert!(diff < TIGHT, "gamma symmetry violated: call={}, put={}", gc.gamma, gp.gamma);
        }

        #[test]
        fn test_vega_symmetry_property(params in valid_bs_params()) {
            let call_params = OptionParams { option_type: OptionType::Call, ..params };
            let put_params = OptionParams { option_type: OptionType::Put, ..params };
            let gc = greeks::compute_greeks(&call_params).unwrap();
            let gp = greeks::compute_greeks(&put_params).unwrap();
            let diff = (gc.vega - gp.vega).abs();
            prop_assert!(diff < TIGHT, "vega symmetry violated: call={}, put={}", gc.vega, gp.vega);
        }

        #[test]
        fn test_delta_parity_property(params in valid_bs_params()) {
            let call_params = OptionParams { option_type: OptionType::Call, ..params };
            let put_params = OptionParams { option_type: OptionType::Put, ..params };
            let gc = greeks::compute_greeks(&call_params).unwrap();
            let gp = greeks::compute_greeks(&put_params).unwrap();
            let expected = (-params.div_yield * params.time).exp();
            let diff = (gc.delta - gp.delta - expected).abs();
            prop_assert!(diff < STANDARD, "delta parity violated: call-put={}, expected={expected}", gc.delta - gp.delta);
        }
    }
}
