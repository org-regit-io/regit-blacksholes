// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Pricing models for European options.
//!
//! Each submodule implements exactly one pricing model from its primary
//! paper source. All functions are pure: parameters in, price out.
//!
//! - [`black_scholes`] — Vanilla European with continuous dividend (Merton 1973)
//! - [`black76`] — Options on futures/forwards (Black 1976)
//! - [`bachelier`] — Normal model for rates near/below zero (Bachelier 1900)
//! - [`displaced`] — Shifted log-normal (Rubinstein 1983)

pub mod bachelier;
pub mod black76;
pub mod black_scholes;
pub mod displaced;
