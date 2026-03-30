// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for regit-blackscholes.
//!
//! Performance targets (PRD):
//!
//! | Operation               | Target   |
//! |-------------------------|----------|
//! | Normal CDF (f64)        | < 5 ns   |
//! | Vanilla BS price (f64)  | < 15 ns  |
//! | Full 17 Greeks (f64)    | < 80 ns  |
//! | IV solve — standard     | < 150 ns |
//! | IV solve — Jackel path  | < 300 ns |

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use regit_blackscholes::greeks;
use regit_blackscholes::iv::{implied_vol, IvSolver};
use regit_blackscholes::math::{ncdf, npdf};
use regit_blackscholes::models::bachelier::{self, BachelierParams};
use regit_blackscholes::models::black76::{self, Black76Params};
use regit_blackscholes::models::black_scholes;
use regit_blackscholes::types::{OptionParams, OptionType};

// ─── Baseline parameters ────────────────────────────────────────────────────

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
        spot: 100.0_f64,
        strike: 100.0_f64,
        rate: 0.05_f64,
        div_yield: 0.02_f64,
        vol: 0.20_f64,
        time: 1.0_f64,
    }
}

fn deep_otm_call() -> OptionParams<f64> {
    OptionParams {
        option_type: OptionType::Call,
        spot: 100.0_f64,
        strike: 200.0_f64,
        rate: 0.05_f64,
        div_yield: 0.02_f64,
        vol: 0.20_f64,
        time: 1.0_f64,
    }
}

// ─── Math benchmarks ────────────────────────────────────────────────────────

fn bench_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("math");

    // Target: < 5 ns
    group.bench_function("ncdf", |b| {
        b.iter(|| ncdf(black_box(0.25_f64)))
    });

    group.bench_function("npdf", |b| {
        b.iter(|| npdf(black_box(0.25_f64)))
    });

    group.finish();
}

// ─── Pricing benchmarks ─────────────────────────────────────────────────────

fn bench_pricing(c: &mut Criterion) {
    let mut group = c.benchmark_group("pricing");

    // Target: < 15 ns
    group.bench_function("bs_call_price", |b| {
        let params = atm_call();
        b.iter(|| black_scholes::price(black_box(&params)))
    });

    group.bench_function("bs_put_price", |b| {
        let params = atm_put();
        b.iter(|| black_scholes::price(black_box(&params)))
    });

    group.bench_function("black76_price", |b| {
        let params = Black76Params {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        b.iter(|| black76::price(black_box(&params)))
    });

    group.bench_function("bachelier_price", |b| {
        let params = BachelierParams {
            option_type: OptionType::Call,
            forward: 100.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            normal_vol: 5.0_f64,
            time: 1.0_f64,
        };
        b.iter(|| bachelier::price(black_box(&params)))
    });

    group.finish();
}

// ─── Greeks benchmarks ──────────────────────────────────────────────────────

fn bench_greeks(c: &mut Criterion) {
    let mut group = c.benchmark_group("greeks");

    // Target: < 80 ns
    group.bench_function("greeks_call", |b| {
        let params = atm_call();
        b.iter(|| greeks::compute_greeks(black_box(&params)))
    });

    group.bench_function("greeks_put", |b| {
        let params = atm_put();
        b.iter(|| greeks::compute_greeks(black_box(&params)))
    });

    group.finish();
}

// ─── Implied volatility benchmarks ──────────────────────────────────────────

fn bench_iv(c: &mut Criterion) {
    let mut group = c.benchmark_group("iv");

    // Pre-compute market prices for IV recovery
    let atm_params = atm_call();
    let atm_market_price = black_scholes::price(&atm_params).unwrap();

    let deep_otm_params = deep_otm_call();
    let deep_otm_market_price = black_scholes::price(&deep_otm_params).unwrap();

    // IV input params: vol field is ignored by the solver (it recovers vol from price)
    let atm_iv_params = atm_call();
    let deep_otm_iv_params = deep_otm_call();

    // Target: < 150 ns (standard ATM regime)
    group.bench_function("iv_atm_auto", |b| {
        b.iter(|| {
            implied_vol(
                black_box(&atm_iv_params),
                black_box(atm_market_price),
                IvSolver::Auto,
            )
        })
    });

    // Target: < 300 ns (Jackel / deep OTM path)
    group.bench_function("iv_deep_otm_auto", |b| {
        b.iter(|| {
            implied_vol(
                black_box(&deep_otm_iv_params),
                black_box(deep_otm_market_price),
                IvSolver::Auto,
            )
        })
    });

    group.bench_function("iv_atm_halley", |b| {
        b.iter(|| {
            implied_vol(
                black_box(&atm_iv_params),
                black_box(atm_market_price),
                IvSolver::Halley,
            )
        })
    });

    group.bench_function("iv_atm_newton", |b| {
        b.iter(|| {
            implied_vol(
                black_box(&atm_iv_params),
                black_box(atm_market_price),
                IvSolver::Newton,
            )
        })
    });

    group.bench_function("iv_atm_brent", |b| {
        b.iter(|| {
            implied_vol(
                black_box(&atm_iv_params),
                black_box(atm_market_price),
                IvSolver::Brent,
            )
        })
    });

    group.finish();
}

// ─── Criterion harness ──────────────────────────────────────────────────────

criterion_group!(benches, bench_math, bench_pricing, bench_greeks, bench_iv);
criterion_main!(benches);
