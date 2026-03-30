# regit-blackscholes

Zero-dependency Black-Scholes options pricing engine. Pure Rust.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

## What it does

`regit-blackscholes` takes an option's market parameters and returns: price, all 17 Greeks through 3rd order, and implied volatility. Covers four models used in European regulated markets. Ships with no external math dependencies.

Every algorithm is hand-rolled from primary paper sources. A regulator, quant auditor, or new engineer can open any source file and trace every number to a citable formula.

## Quick start

```toml
[dependencies]
regit-blackscholes = "1.0"
```

```rust
use regit_blackscholes::{OptionParams, OptionType, Pricing, GreeksCalc, ImpliedVol, IvSolver};

let params = OptionParams {
    option_type: OptionType::Call,
    spot:        100.0_f64,
    strike:      100.0_f64,
    rate:        0.05_f64,
    div_yield:   0.02_f64,
    vol:         0.20_f64,
    time:        1.0_f64,
};

// Price
let price = params.price().unwrap();
println!("Price: {price:.4}");  // 9.2270

// All 17 Greeks in one call
let greeks = params.greeks().unwrap();
println!("Delta: {:.4}, Gamma: {:.6}, Vega: {:.4}", greeks.delta, greeks.gamma, greeks.vega);

// Implied volatility from market price
let iv = params.implied_vol(price, IvSolver::Auto).unwrap();
println!("IV: {iv:.6}");  // 0.200000
```

See [`examples/quickstart.rs`](examples/quickstart.rs) for a complete working example covering all four models.

## Models

| Model | Use case | Reference |
|-------|----------|-----------|
| Black-Scholes-Merton | Vanilla European, continuous dividend | Merton (1973) |
| Black-76 | Futures, forwards, caps/floors, swaptions | Black (1976) |
| Bachelier | Rates options near/below zero (normal vol) | Bachelier (1900) |
| Displaced log-normal | SABR calibration, bridges Black-76 and Bachelier | Rubinstein (1983) |

```rust
use regit_blackscholes::{Model, Black76Params, BachelierParams, DisplacedParams, Pricing};

// All models share the same Pricing trait
let models = vec![
    Model::BlackScholes(params),
    Model::Black76(Black76Params { option_type, forward: 100.0, strike: 100.0, rate: 0.05, vol: 0.20, time: 1.0 }),
    Model::Bachelier(BachelierParams { option_type, forward: 100.0, strike: 100.0, rate: 0.05, normal_vol: 5.0, time: 1.0 }),
    Model::Displaced(DisplacedParams { option_type, forward: 100.0, strike: 100.0, rate: 0.05, vol: 0.20, time: 1.0, displacement: 50.0 }),
];

for model in &models {
    println!("{:.4}", model.price().unwrap());
}
```

## Greeks

All 17 analytic Greeks computed in a single call from shared intermediates. No finite differences. No recomputation.

| Order | Greeks |
|-------|--------|
| 1st | Delta, Theta, Vega, Rho, Epsilon, Lambda, Dual Delta |
| 2nd | Gamma, Vanna, Charm, Veta, Vomma, Dual Gamma |
| 3rd | Speed, Zomma, Color, Ultima |

```rust
let greeks = params.greeks().unwrap();

// First order
greeks.delta;       // dV/dS
greeks.vega;        // dV/dsigma
greeks.theta;       // dV/dt (daily)
greeks.rho;         // dV/dr

// Second order
greeks.gamma;       // d2V/dS2
greeks.vanna;       // d2V/dS dsigma
greeks.vomma;       // d2V/dsigma2

// Third order
greeks.speed;       // d3V/dS3
greeks.zomma;       // d3V/dS2 dsigma
greeks.ultima;      // d3V/dsigma3
```

## Implied volatility

Multi-strategy solver chain with automatic fallback:

| Step | Algorithm | When | Convergence |
|------|-----------|------|-------------|
| Initial guess | Corrado-Miller (1996) | Always | Closed-form |
| Primary | Halley's method | Vega > 1e-12 | 3rd order (cubic) |
| Fallback | Newton-Raphson | Halley unstable | 2nd order (quadratic) |
| Deep OTM/ITM | Jackel-inspired | Near-zero vega | Rational approx + refinement |
| Last resort | Brent's method | All else fails | Guaranteed (bracketed) |

```rust
// Automatic strategy selection (recommended)
let iv = params.implied_vol(market_price, IvSolver::Auto)?;

// Or force a specific solver
let iv = params.implied_vol(market_price, IvSolver::Halley)?;
let iv = params.implied_vol(market_price, IvSolver::Brent)?;
```

Convergence tolerance: `|sigma_new - sigma_old| < 1e-10`. Native Rust — no C++ FFI, no `cc` build dependency.

## Architecture

```
src/
  lib.rs                    # Module declarations + re-exports
  types.rs                  # OptionParams, Greeks, Pricing/GreeksCalc/ImpliedVol traits, Model enum
  errors.rs                 # PricingError, IvError
  math.rs                   # Normal CDF/PDF (Horner rational approx), d1/d2

  models/
    black_scholes.rs        # Merton 1973 continuous-dividend
    black76.rs              # Black 1976 futures/forwards
    bachelier.rs            # Normal model (rates)
    displaced.rs            # Shifted log-normal (Rubinstein 1983)

  greeks.rs                 # All 17 analytic Greeks
  iv.rs                     # IV solver chain
```

One file, one domain. Each function is pure and composable.

## Testing

```bash
cargo test                        # 336 tests
cargo run --example quickstart    # Library usage demo
cargo bench                       # Criterion benchmarks
```

**203 unit tests** covering golden values, edge cases, error paths, and formula consistency across all four models and the IV solver chain.

**109 integration tests** across 5 suites:
- `golden` — regression anchors against QuantLib reference values
- `parity` — put-call parity for all four models
- `greeks_tests` — cross-Greek relationships, finite-difference verification, homogeneity
- `boundaries` — T=0, sigma=0, deep OTM/ITM, negative rates
- `properties` — proptest invariants (1,000 cases each): delta bounds, gamma/vega positivity, price monotonicity

**24 doc tests** — every public function's example compiles and runs.

## Code quality

- `#![forbid(unsafe_code)]` crate-wide
- `clippy::pedantic` with zero warnings
- Every public function documented with mathematical references
- No `unwrap()` or `panic!()` in library code
- Deterministic: same input produces bit-identical output
- All polynomial evaluation via `f64::mul_add` (Horner's method)

## Dependencies

**Runtime: zero.** Only `std`. No `statrs`, no `libm`, no `nalgebra`, no `cc` FFI.

| Crate | Purpose | Scope | License |
|-------|---------|-------|---------|
| `wide` | SIMD batch pricing | Optional (`simd` feature) | Zlib |
| `criterion` | Benchmarks | Dev only | Apache-2.0/MIT |
| `proptest` | Property testing | Dev only | Apache-2.0/MIT |
| `approx` | Float comparison | Dev only | Apache-2.0 |

License policy enforced via `cargo-deny`. No copyleft dependencies.

## Algorithms

All implemented from primary paper sources. No ports from Python, no reading existing Rust crates.

| Algorithm | Reference |
|-----------|-----------|
| Normal CDF rational approximation | Abramowitz & Stegun, *Handbook of Mathematical Functions*, section 26.2.17 (1964) |
| Black-Scholes-Merton pricing | Black & Scholes (1973); Merton (1973) |
| Black-76 futures pricing | Black, *Journal of Financial Economics* (1976) |
| Bachelier normal model | Bachelier (1900); Schachermayer & Teichmann (2008) |
| Displaced log-normal | Rubinstein, *Journal of Finance* (1983) |
| Corrado-Miller IV guess | Corrado & Miller, *Journal of Financial Economics* (1996) |
| Jackel "Let's Be Rational" | Jackel, *Wilmott Magazine* (2016) |
| Brent's method | Brent, *Algorithms for Minimization Without Derivatives* (1973) |

## Documentation

- [MATH.md](MATH.md) — Full mathematical derivations for every algorithm
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [SECURITY.md](SECURITY.md) — Vulnerability disclosure policy

## License

Apache License 2.0. See [LICENSE](LICENSE).

```
Copyright 2026 Regit.io — Nicolas Koenig
```

---

Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.
