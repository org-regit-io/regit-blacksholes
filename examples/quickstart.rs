// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Quickstart example — ergonomic trait-based API.
//!
//! Demonstrates pricing, Greeks, implied volatility, and model dispatch.

use regit_blackscholes::{
    BachelierParams, Black76Params, DisplacedParams, GreeksCalc, ImpliedVol, IvSolver, Model,
    OptionParams, OptionType, Pricing,
};

fn main() {
    // ── Black-Scholes-Merton ────────────────────────────────────────────
    let params = OptionParams {
        option_type: OptionType::Call,
        spot: 100.0_f64,
        strike: 100.0_f64,
        rate: 0.05_f64,
        div_yield: 0.02_f64,
        vol: 0.20_f64,
        time: 1.0_f64,
    };

    let price = params.price().unwrap();
    println!("BS call price: {price:.4}");

    let greeks = params.greeks().unwrap();
    println!("  delta: {:.4}", greeks.delta);
    println!("  gamma: {:.6}", greeks.gamma);
    println!("  vega:  {:.4}", greeks.vega);

    // Implied vol roundtrip
    let iv_params = OptionParams {
        vol: 0.0_f64,
        ..params
    };
    let iv = iv_params.implied_vol(price, IvSolver::Auto).unwrap();
    println!("  implied vol: {iv:.6}");

    // ── Black-76 ────────────────────────────────────────────────────────
    let b76 = Black76Params {
        option_type: OptionType::Call,
        forward: 100.0_f64,
        strike: 100.0_f64,
        rate: 0.05_f64,
        vol: 0.20_f64,
        time: 1.0_f64,
    };
    println!("\nBlack-76 call: {:.4}", b76.price().unwrap());

    // ── Bachelier ───────────────────────────────────────────────────────
    let bach = BachelierParams {
        option_type: OptionType::Call,
        forward: 100.0_f64,
        strike: 100.0_f64,
        rate: 0.05_f64,
        normal_vol: 5.0_f64,
        time: 1.0_f64,
    };
    println!("Bachelier call: {:.4}", bach.price().unwrap());

    // ── Displaced log-normal ────────────────────────────────────────────
    let disp = DisplacedParams {
        option_type: OptionType::Call,
        forward: 100.0_f64,
        strike: 100.0_f64,
        rate: 0.05_f64,
        vol: 0.20_f64,
        time: 1.0_f64,
        displacement: 50.0_f64,
    };
    println!("Displaced call: {:.4}", disp.price().unwrap());

    // ── Model enum dispatch ─────────────────────────────────────────────
    let models: Vec<Model> = vec![
        Model::BlackScholes(params),
        Model::Black76(b76),
        Model::Bachelier(bach),
        Model::Displaced(disp),
    ];
    println!("\nModel enum prices:");
    for m in &models {
        println!("  {m:?} -> {:.4}", m.price().unwrap());
    }
}
