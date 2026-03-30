// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Zero-dependency Black-Scholes options pricing engine in pure Rust.
//!
//! Covers four European pricing models — Black-Scholes-Merton, Black-76,
//! Bachelier, and Displaced Diffusion — with all 17 analytic Greeks through
//! 3rd order and a multi-strategy implied volatility solver.
//!
//! Designed for auditability: every algorithm is hand-rolled from primary
//! paper sources with no external math dependencies. A regulator, quant
//! auditor, or new engineer can trace every number to a citable formula.
//!
//! # Models
//!
//! - [`models::black_scholes`] — Vanilla European, continuous dividend (Merton 1973)
//! - [`models::black76`] — Futures/forwards (Black 1976)
//! - [`models::bachelier`] — Normal model for rates near/below zero
//! - [`models::displaced`] — Shifted log-normal (Rubinstein 1983)
//!
//! # Architecture
//!
//! ```text
//! types/errors → math primitives (ncdf, npdf) → pricing models
//!                                              → greeks (analytic, 17 total)
//!                                              → implied volatility (solver chain)
//!                                              → batch SIMD (feature-gated)
//! ```
//!
//! Part of [Regit OS](https://www.regit.io) — the operating system for
//! investment products. From Luxembourg.

pub mod errors;
pub mod greeks;
pub mod iv;
pub mod math;
pub mod models;
pub mod types;

#[cfg(feature = "simd")]
pub mod batch;

// ─── Re-exports for ergonomic top-level access ─────────────────────────────

pub use errors::{IvError, PricingError};
pub use iv::IvSolver;
pub use models::bachelier::BachelierParams;
pub use models::black76::Black76Params;
pub use models::displaced::DisplacedParams;
pub use types::{
    Float, Greeks, GreeksCalc, ImpliedVol, Model, OptionParams, OptionType, Pricing,
};
