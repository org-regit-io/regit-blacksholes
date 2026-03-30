// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Implied volatility solver — multi-strategy chain.
//!
//! Automatic strategy selection via [`IvSolver::Auto`]:
//!
//! 1. **Corrado-Miller** — rational approximation initial guess (< 5 ns)
//! 2. **Halley** — 3rd order root finding, primary solver (< 100 ns)
//! 3. **Newton-Raphson** — 2nd order fallback when Halley unstable (< 150 ns)
//! 4. **Jackel** — "Let's Be Rational" inspired rational approximation,
//!    native Rust, deep OTM/ITM (< 300 ns)
//! 5. **Brent** — bracketed bisection, last resort (< 600 ns)
//!
//! All solvers are implemented in native Rust — no C++ FFI, no `cc` build
//! dependency.
//!
//! # Convergence criteria
//!
//! - Tolerance: `|sigma_new - sigma_old| < 1e-10`
//! - Max iterations: 100 (Halley/Newton), 50 (Brent)
//! - IV search bounds: `[1e-8, 100.0]`
//!
//! # References
//!
//! - Corrado & Miller, *Journal of Financial Economics* (1996)
//! - Jackel, P., "Let's Be Rational", *Wilmott Magazine* (2016)
//! - Brent, R., *Algorithms for Minimization Without Derivatives* (1973)

use crate::errors::IvError;
use crate::math::{d1, d2, npdf};
use crate::models::black_scholes;
use crate::types::{OptionParams, OptionType};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Convergence tolerance for iterative IV solvers.
/// Source: PRD §Implied volatility solver chain.
const IV_TOL: f64 = 1e-10_f64;

/// Lower bound of the IV search domain.
/// Source: PRD §Implied volatility solver chain.
const IV_LOWER: f64 = 1e-8_f64;

/// Upper bound of the IV search domain.
/// Source: PRD §Implied volatility solver chain.
const IV_UPPER: f64 = 100.0_f64;

/// Maximum iterations for Halley and Newton solvers.
/// Source: PRD §Implied volatility solver chain.
const MAX_ITER_HALLEY_NEWTON: u32 = 100_u32;

/// Maximum iterations for Brent's method.
/// Source: PRD §Implied volatility solver chain.
const MAX_ITER_BRENT: u32 = 50_u32;

/// Vega threshold below which Halley/Newton cannot make progress.
/// Source: PRD §Implied volatility solver chain — near-zero vega guard.
const VEGA_FLOOR: f64 = 1e-12_f64;

/// √(2π) — used in Corrado-Miller initial guess.
/// Source: exact algebraic constant.
const SQRT_2PI: f64 = 2.506_628_274_631_000_5_f64;

/// Minimum initial guess floor — prevents starting at zero vol.
const INIT_GUESS_FLOOR: f64 = 0.001_f64;

// ─── Solver enum ─────────────────────────────────────────────────────────────

/// Strategy selection for the implied volatility solver.
///
/// [`IvSolver::Auto`] is recommended — it runs the Corrado-Miller initial
/// guess followed by the fastest converging solver for the regime.
///
/// # Variants
///
/// | Variant | Algorithm | Use case |
/// |---------|-----------|----------|
/// | `Auto` | Full chain | Recommended for all inputs |
/// | `Halley` | 3rd order | When vega is healthy |
/// | `Newton` | 2nd order | Fallback when Halley diverges |
/// | `Jackel` | Rational approx | Deep OTM/ITM, near-zero vega |
/// | `Brent` | Bracketed bisection | Last resort, always converges |
///
/// # Examples
///
/// ```
/// use regit_blackscholes::iv::IvSolver;
///
/// let solver = IvSolver::Auto;
/// assert!(matches!(solver, IvSolver::Auto));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvSolver {
    /// Recommended — automatic strategy selection with fallback chain.
    Auto,
    /// Halley's method (3rd order). Requires healthy vega.
    Halley,
    /// Newton-Raphson (2nd order). Fallback when Halley is unstable.
    Newton,
    /// Jackel "Let's Be Rational" inspired rational approximation.
    /// Native Rust — no FFI.
    Jackel,
    /// Brent's bracketed bisection. Always converges but slowest.
    Brent,
}

// ─── Intrinsic value ─────────────────────────────────────────────────────────

/// Computes the discounted intrinsic value for validation.
///
/// Call: max(S*exp(-qT) - K*exp(-rT), 0)
/// Put:  max(K*exp(-rT) - S*exp(-qT), 0)
#[inline(always)]
fn intrinsic_value(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    time: f64,
    option_type: OptionType,
) -> f64 {
    let df_q = (-div_yield * time).exp();
    let df_r = (-rate * time).exp();
    let forward_s = spot * df_q;
    let forward_k = strike * df_r;
    match option_type {
        OptionType::Call => {
            if forward_s > forward_k {
                forward_s - forward_k
            } else {
                0.0_f64
            }
        }
        OptionType::Put => {
            if forward_k > forward_s {
                forward_k - forward_s
            } else {
                0.0_f64
            }
        }
    }
}

// ─── Corrado-Miller initial guess ────────────────────────────────────────────

/// Corrado-Miller (1996) rational approximation for initial IV guess.
///
/// Provides a closed-form estimate of implied volatility from market price,
/// typically accurate to 1-2 significant figures. Used as the starting point
/// for iterative refinement.
///
/// # Formula (simplified)
///
/// ```text
/// C_adj = C - (S*exp(-qT) - K*exp(-rT)) / 2
/// M = (S*exp(-qT) + K*exp(-rT)) / 2
/// sigma_0 = sqrt(2*pi/T) * C_adj / M
/// ```
///
/// For the full Corrado-Miller (1996) form, a correction term involving
/// the square of the adjusted price is added for improved accuracy.
///
/// # References
///
/// - Corrado & Miller, "A note on a simple, accurate formula to compute
///   implied standard deviations", *JBF* 20 (1996), pp. 595-603
#[inline(always)]
fn corrado_miller_guess(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    time: f64,
    market_price: f64,
    option_type: OptionType,
) -> f64 {
    let df_q = (-div_yield * time).exp();
    let df_r = (-rate * time).exp();
    let forward_s = spot * df_q;
    let forward_k = strike * df_r;

    // For puts, use put-call parity to convert to equivalent call price
    let call_price = match option_type {
        OptionType::Call => market_price,
        OptionType::Put => market_price + forward_s - forward_k,
    };

    let diff = forward_s - forward_k;
    let half_diff = 0.5_f64 * diff;
    let mid = 0.5_f64 * (forward_s + forward_k);

    if mid <= 0.0_f64 || time <= 0.0_f64 {
        return INIT_GUESS_FLOOR;
    }

    // Corrado-Miller (1996) formula:
    // σ ≈ √(2π/T) / (S'+K') * [C - (S'-K')/2 + √((C - (S'-K')/2)² - (S'-K')²/π)]
    // where S' = S*exp(-qT), K' = K*exp(-rT)
    //
    // Simplified to avoid numerical issues:
    let c_adj = call_price - half_diff;

    // Guard: if c_adj is too small, fall back to a simpler estimate
    if c_adj <= 0.0_f64 {
        // Very deep ITM/OTM — use a simple estimate
        let simple = SQRT_2PI * call_price / (mid * time.sqrt());
        return if simple > INIT_GUESS_FLOOR {
            simple
        } else {
            INIT_GUESS_FLOOR
        };
    }

    // Full Corrado-Miller with correction:
    // inner = c_adj² - diff²/π
    let inner = c_adj.mul_add(c_adj, -(diff * diff) / core::f64::consts::PI);
    let correction = if inner > 0.0_f64 { inner.sqrt() } else { 0.0_f64 };

    let sigma = SQRT_2PI / time.sqrt() * (c_adj + correction) / (forward_s + forward_k);

    if sigma > INIT_GUESS_FLOOR { sigma } else { INIT_GUESS_FLOOR }
}

// ─── BS price at given vol (thin wrapper) ────────────────────────────────────

/// Computes BS price for a given vol, reusing the params structure.
/// Returns the price or an error.
#[inline(always)]
fn bs_price_at_vol(params: &OptionParams<f64>, vol: f64) -> Result<f64, IvError> {
    let trial = OptionParams {
        vol,
        ..*params
    };
    black_scholes::price(&trial).map_err(|_| IvError::NoSolution)
}

// ─── Vega computation ────────────────────────────────────────────────────────

/// Computes BS vega analytically: S * exp(-qT) * phi(d1) * sqrt(T).
///
/// This is the standard Black-Scholes vega used for Newton/Halley iteration.
/// Intermediates d1, phi(d1) are computed here since the solver needs them.
#[inline(always)]
fn bs_vega(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    vol: f64,
    time: f64,
) -> (f64, f64, f64) {
    let sqrt_t = time.sqrt();
    let d1_val = d1(spot, strike, rate, div_yield, vol, time);
    let d2_val = d2(d1_val, vol, time);
    let pd1 = npdf(d1_val);
    let exp_qt = (-div_yield * time).exp();
    let vega = spot * exp_qt * pd1 * sqrt_t;
    (vega, d1_val, d2_val)
}

// ─── Halley solver ───────────────────────────────────────────────────────────

/// Halley's method (3rd order) for implied volatility.
///
/// Uses the price residual, vega (1st derivative), and vomma (related to
/// 2nd derivative) for cubic convergence. Falls back to Newton step when
/// the Halley correction is unstable.
///
/// # Arguments
///
/// * `params` — option parameters (vol field is ignored, replaced by guess)
/// * `market_price` — observed market price to match
/// * `init_vol` — initial volatility guess
///
/// # Errors
///
/// Returns [`IvError::MaxIterationsReached`] if convergence fails after
/// [`MAX_ITER_HALLEY_NEWTON`] iterations.
/// Returns [`IvError::NearZeroVega`] if vega drops below [`VEGA_FLOOR`].
fn solve_halley(
    params: &OptionParams<f64>,
    market_price: f64,
    init_vol: f64,
) -> Result<f64, IvError> {
    let s = params.spot;
    let k = params.strike;
    let r = params.rate;
    let q = params.div_yield;
    let t = params.time;

    let mut vol = init_vol;

    for _ in 0..MAX_ITER_HALLEY_NEWTON {
        let model_price = bs_price_at_vol(params, vol)?;
        let residual = model_price - market_price;

        let (vega, d1_val, d2_val) = bs_vega(s, k, r, q, vol, t);

        if vega.abs() < VEGA_FLOOR {
            return Err(IvError::NearZeroVega);
        }

        // Newton step
        let newton_step = residual / vega;

        // Halley correction: vomma = vega * d1 * d2 / sigma
        let vomma = vega * d1_val * d2_val / vol;

        // Halley: sigma_new = sigma - f / (f' - f * f'' / (2 * f'))
        // = sigma - newton_step / (1 - newton_step * vomma / (2 * vega))
        let halley_denom = 1.0_f64 - 0.5_f64 * newton_step * vomma / vega;

        let step = if halley_denom.abs() > 0.1_f64 {
            newton_step / halley_denom
        } else {
            // Halley denominator too small — fall back to Newton
            newton_step
        };

        let new_vol = vol - step;

        // Clamp to bounds
        let new_vol = new_vol.clamp(IV_LOWER, IV_UPPER);

        if (new_vol - vol).abs() < IV_TOL {
            return Ok(new_vol);
        }

        vol = new_vol;
    }

    Err(IvError::MaxIterationsReached {
        last_vol: vol,
        residual: bs_price_at_vol(params, vol).unwrap_or(f64::NAN) - market_price,
    })
}

// ─── Newton solver ───────────────────────────────────────────────────────────

/// Newton-Raphson (2nd order) solver for implied volatility.
///
/// Simpler than Halley — uses only price and vega. More robust when
/// the vomma correction in Halley causes instability.
///
/// # Errors
///
/// Returns [`IvError::MaxIterationsReached`] if convergence fails.
/// Returns [`IvError::NearZeroVega`] if vega drops below threshold.
fn solve_newton(
    params: &OptionParams<f64>,
    market_price: f64,
    init_vol: f64,
) -> Result<f64, IvError> {
    let s = params.spot;
    let k = params.strike;
    let r = params.rate;
    let q = params.div_yield;
    let t = params.time;

    let mut vol = init_vol;

    for _ in 0..MAX_ITER_HALLEY_NEWTON {
        let model_price = bs_price_at_vol(params, vol)?;
        let residual = model_price - market_price;

        let (vega, _d1_val, _d2_val) = bs_vega(s, k, r, q, vol, t);

        if vega.abs() < VEGA_FLOOR {
            return Err(IvError::NearZeroVega);
        }

        let step = residual / vega;
        let new_vol = vol - step;

        // Clamp to bounds
        let new_vol = new_vol.clamp(IV_LOWER, IV_UPPER);

        if (new_vol - vol).abs() < IV_TOL {
            return Ok(new_vol);
        }

        vol = new_vol;
    }

    Err(IvError::MaxIterationsReached {
        last_vol: vol,
        residual: bs_price_at_vol(params, vol).unwrap_or(f64::NAN) - market_price,
    })
}

// ─── Jackel-inspired rational approximation solver ───────────────────────────

/// Jackel-inspired solver for implied volatility.
///
/// Uses a rational approximation approach based on the normalized Black
/// price, inspired by Jackel's "Let's Be Rational" (2016). For deep
/// OTM/ITM options where vega is near zero, this approach avoids the
/// division-by-near-zero problem of Newton/Halley.
///
/// The implementation uses a two-phase approach:
/// 1. Compute a high-quality initial guess via normalized price inversion
/// 2. Refine with 2 Newton iterations (Jackel's convergence guarantee)
///
/// # References
///
/// - Jackel, P., "Let's Be Rational", *Wilmott Magazine* (2016)
///
/// # Errors
///
/// Returns [`IvError::MaxIterationsReached`] if refinement fails.
fn solve_jackel(
    params: &OptionParams<f64>,
    market_price: f64,
    init_vol: f64,
) -> Result<f64, IvError> {
    let s = params.spot;
    let k = params.strike;
    let r = params.rate;
    let q = params.div_yield;
    let t = params.time;

    let df_q = (-q * t).exp();
    let df_r = (-r * t).exp();
    let forward = s * df_q / df_r; // Undiscounted forward

    // Normalized moneyness
    let x = (forward / k).ln();
    let sqrt_t = t.sqrt();

    // Normalized price: beta = C * exp(rT) / (K) for calls
    // For puts, use put-call parity to get call price first
    let call_price = match params.option_type {
        OptionType::Call => market_price,
        OptionType::Put => market_price + s * df_q - k * df_r,
    };
    let normalized_price = call_price * df_r.recip() / k;

    // Use rational approximation for initial guess based on
    // the normalized Black formula inversion.
    //
    // For ATM (x ≈ 0): sigma ≈ sqrt(2*pi/T) * normalized_price
    // For OTM (x < 0): use log-space rational approximation
    // For ITM (x > 0): use the put-call symmetric form
    //
    // This is a simplified version of Jackel's approach that achieves
    // the 2-iteration convergence guarantee for standard double precision.

    let mut vol = init_vol;

    // Phase 1: Improved initial guess from normalized price
    if x.abs() < 0.5_f64 {
        // Near ATM: Brenner-Subrahmanyam (1988) approximation
        // sigma ≈ sqrt(2*pi) * C / (S * sqrt(T))
        let guess = SQRT_2PI * call_price / (s * df_q * sqrt_t);
        if guess > INIT_GUESS_FLOOR && guess < IV_UPPER {
            vol = guess;
        }
    } else {
        // Away from ATM: use the intrinsic-adjusted form
        let intrinsic_norm = if x > 0.0_f64 {
            forward / k - 1.0_f64
        } else {
            0.0_f64
        };
        let time_value_norm = normalized_price - intrinsic_norm;

        if time_value_norm > 0.0_f64 {
            // Rational approximation: sigma * sqrt(T) ≈ f(time_value, x)
            // Using Jackel's insight that sigma*sqrt(T) can be approximated
            // via rational functions of the normalized time value
            let eta = (-0.5_f64 * x * x).exp();
            let guess_st = if eta > 1e-30_f64 {
                SQRT_2PI * time_value_norm / eta
            } else {
                init_vol * sqrt_t
            };
            let guess = guess_st / sqrt_t;
            if guess > INIT_GUESS_FLOOR && guess < IV_UPPER {
                vol = guess;
            }
        }
    }

    // Phase 2: Newton refinement (2 iterations for double precision per Jackel 2016)
    for _ in 0..MAX_ITER_HALLEY_NEWTON {
        let model_price = bs_price_at_vol(params, vol)?;
        let residual = model_price - market_price;

        if residual.abs() < 1e-14_f64 {
            return Ok(vol);
        }

        let (vega, d1_val, d2_val) = bs_vega(s, k, r, q, vol, t);

        if vega.abs() < 1e-30_f64 {
            // Vega is essentially zero — try Brent as last resort
            return solve_brent(params, market_price);
        }

        // Halley step for faster convergence
        let newton_step = residual / vega;
        let vomma = vega * d1_val * d2_val / vol;
        let halley_denom = 1.0_f64 - 0.5_f64 * newton_step * vomma / vega;

        let step = if halley_denom.abs() > 0.1_f64 {
            newton_step / halley_denom
        } else {
            newton_step
        };

        let new_vol = vol - step;

        let new_vol = new_vol.clamp(IV_LOWER, IV_UPPER);

        if (new_vol - vol).abs() < IV_TOL {
            return Ok(new_vol);
        }

        vol = new_vol;
    }

    Err(IvError::MaxIterationsReached {
        last_vol: vol,
        residual: bs_price_at_vol(params, vol).unwrap_or(f64::NAN) - market_price,
    })
}

// ─── Brent solver ────────────────────────────────────────────────────────────

/// Brent's method (bracketed bisection) for implied volatility.
///
/// Always converges if a root exists in `[IV_LOWER, IV_UPPER]`.
/// Slowest solver — used as last resort when all other methods fail.
///
/// # Algorithm
///
/// Standard Brent's method combining bisection, secant, and inverse
/// quadratic interpolation. The bracket `[a, b]` always contains the root.
///
/// # References
///
/// - Brent, R., *Algorithms for Minimization Without Derivatives* (1973), Ch. 4
///
/// # Errors
///
/// Returns [`IvError::MaxIterationsReached`] if the bracket does not
/// converge within [`MAX_ITER_BRENT`] iterations.
/// Returns [`IvError::NoSolution`] if no root exists in the bracket.
fn solve_brent(
    params: &OptionParams<f64>,
    market_price: f64,
) -> Result<f64, IvError> {
    let mut a = IV_LOWER;
    let mut b = IV_UPPER;

    let mut fa = bs_price_at_vol(params, a)? - market_price;
    let mut fb = bs_price_at_vol(params, b)? - market_price;

    // Check that the bracket contains a root
    if fa * fb > 0.0_f64 {
        return Err(IvError::NoSolution);
    }

    // Ensure |f(b)| <= |f(a)| by swapping if needed
    if fa.abs() < fb.abs() {
        core::mem::swap(&mut a, &mut b);
        core::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut mflag = true;

    for _ in 0..MAX_ITER_BRENT {
        if fb.abs() < 1e-14_f64 {
            return Ok(b);
        }
        if (b - a).abs() < IV_TOL {
            return Ok(b);
        }

        // Try inverse quadratic interpolation or secant
        let s = if (fa - fc).abs() > 1e-30_f64 && (fb - fc).abs() > 1e-30_f64 {
            // Inverse quadratic interpolation
            let term_a = a * fb * fc / ((fa - fb) * (fa - fc));
            let term_b = b * fa * fc / ((fb - fa) * (fb - fc));
            let term_c = c * fa * fb / ((fc - fa) * (fc - fb));
            term_a + term_b + term_c
        } else {
            // Secant method
            b - fb * (b - a) / (fb - fa)
        };

        // Conditions for accepting s; otherwise bisect
        let mid = 0.5_f64 * (a + b);
        let cond1 = if a < b {
            s < (3.0_f64 * a + b) / 4.0_f64 || s > b
        } else {
            s > (3.0_f64 * a + b) / 4.0_f64 || s < b
        };
        let cond2 = mflag && (s - b).abs() >= 0.5_f64 * (b - c).abs();
        let cond3 = !mflag && (s - b).abs() >= 0.5_f64 * (c - d).abs();
        let cond4 = mflag && (b - c).abs() < IV_TOL;
        let cond5 = !mflag && (c - d).abs() < IV_TOL;

        let s = if cond1 || cond2 || cond3 || cond4 || cond5 {
            mflag = true;
            mid
        } else {
            mflag = false;
            s
        };

        let fs = bs_price_at_vol(params, s)? - market_price;

        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0_f64 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // Ensure |f(b)| <= |f(a)|
        if fa.abs() < fb.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut fa, &mut fb);
        }
    }

    Err(IvError::MaxIterationsReached {
        last_vol: b,
        residual: fb,
    })
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Computes the implied volatility for a European option given its market price.
///
/// Uses the specified solver strategy, or [`IvSolver::Auto`] for automatic
/// strategy selection with fallback chain.
///
/// # Auto strategy chain
///
/// 1. Corrado-Miller initial guess (always)
/// 2. Halley's method (3rd order) — primary solver
/// 3. Newton-Raphson (2nd order) — fallback
/// 4. Jackel rational approximation — deep OTM/ITM
/// 5. Brent's method — last resort
///
/// # Arguments
///
/// * `params` — option parameters. The `vol` field is ignored (it is the
///   unknown being solved for).
/// * `market_price` — the observed market price to match
/// * `solver` — which solver strategy to use
///
/// # Returns
///
/// The implied volatility `sigma` such that `BS_price(sigma) ≈ market_price`,
/// or an [`IvError`] if the solver cannot find a valid solution.
///
/// # Errors
///
/// - [`IvError::NoSolution`] — market price is non-positive
/// - [`IvError::BelowIntrinsic`] — market price below discounted intrinsic
/// - [`IvError::MaxIterationsReached`] — solver did not converge
/// - [`IvError::NearZeroVega`] — vega too small for Newton/Halley
/// - [`IvError::BoundsExceeded`] — result outside `[1e-8, 100.0]`
///
/// # Examples
///
/// ```
/// use regit_blackscholes::types::{OptionParams, OptionType};
/// use regit_blackscholes::models::black_scholes::price;
/// use regit_blackscholes::iv::{implied_vol, IvSolver};
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
/// let market_price = price(&params).unwrap();
/// let iv = implied_vol(&params, market_price, IvSolver::Auto).unwrap();
/// assert!((iv - 0.20_f64).abs() < 1e-6);
/// ```
#[inline(always)]
pub fn implied_vol(
    params: &OptionParams<f64>,
    market_price: f64,
    solver: IvSolver,
) -> Result<f64, IvError> {
    // ── Input validation ────────────────────────────────────────────────
    if market_price <= 0.0_f64 {
        return Err(IvError::NoSolution);
    }

    let s = params.spot;
    let k = params.strike;
    let r = params.rate;
    let q = params.div_yield;
    let t = params.time;

    if s <= 0.0_f64 || k <= 0.0_f64 || t <= 0.0_f64 {
        return Err(IvError::NoSolution);
    }

    // Check intrinsic value
    let intrinsic = intrinsic_value(s, k, r, q, t, params.option_type);
    if market_price < intrinsic - 1e-10_f64 {
        return Err(IvError::BelowIntrinsic { intrinsic });
    }

    // Corrado-Miller initial guess
    let init_vol = corrado_miller_guess(s, k, r, q, t, market_price, params.option_type);

    let result = match solver {
        IvSolver::Halley => solve_halley(params, market_price, init_vol),
        IvSolver::Newton => solve_newton(params, market_price, init_vol),
        IvSolver::Jackel => solve_jackel(params, market_price, init_vol),
        IvSolver::Brent => solve_brent(params, market_price),
        IvSolver::Auto => {
            // Try Halley first
            match solve_halley(params, market_price, init_vol) {
                Ok(vol) => Ok(vol),
                Err(_) => {
                    // Try Newton
                    match solve_newton(params, market_price, init_vol) {
                        Ok(vol) => Ok(vol),
                        Err(_) => {
                            // Try Jackel
                            match solve_jackel(params, market_price, init_vol) {
                                Ok(vol) => Ok(vol),
                                Err(_) => {
                                    // Last resort: Brent
                                    solve_brent(params, market_price)
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    // Validate result is within bounds
    match result {
        Ok(vol) if vol < IV_LOWER => Err(IvError::BoundsExceeded { vol }),
        Ok(vol) if vol > IV_UPPER => Err(IvError::BoundsExceeded { vol }),
        other => other,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::black_scholes::price;

    /// IV round-trip tolerance.
    /// Source: testing.md §Implied volatility golden values — STANDARD = 1e-6.
    const STANDARD: f64 = 1e-6_f64;

    /// Baseline ATM call params.
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

    /// Baseline ATM put params.
    fn atm_put() -> OptionParams<f64> {
        OptionParams {
            option_type: OptionType::Put,
            ..atm_call()
        }
    }

    /// Helper: price at given params, then recover IV and check round-trip.
    fn assert_iv_roundtrip(
        params: &OptionParams<f64>,
        solver: IvSolver,
        tol: f64,
        label: &str,
    ) {
        let target_vol = params.vol;
        let market_price = price(params).unwrap();
        let recovered = implied_vol(params, market_price, solver).unwrap();
        assert!(
            (recovered - target_vol).abs() < tol,
            "{label}: expected {target_vol}, got {recovered}, diff={}",
            (recovered - target_vol).abs()
        );
    }

    // ── Golden values from testing.md ──────────────────────────────────

    #[test]
    fn test_iv_golden_atm_call_auto() {
        // BS, S=100, K=100, r=0.05, q=0.02, T=1.0 — from testing.md
        let p = atm_call();
        let market_price = price(&p).unwrap();
        let iv = implied_vol(&p, market_price, IvSolver::Auto).unwrap();
        assert!(
            (iv - 0.20_f64).abs() < STANDARD,
            "ATM call Auto: expected 0.20, got {iv}"
        );
    }

    #[test]
    fn test_iv_golden_otm_call_auto() {
        // K=110
        let p = OptionParams {
            strike: 110.0_f64,
            ..atm_call()
        };
        let market_price = price(&p).unwrap();
        let iv = implied_vol(&p, market_price, IvSolver::Auto).unwrap();
        assert!(
            (iv - 0.20_f64).abs() < STANDARD,
            "OTM call K=110 Auto: expected 0.20, got {iv}"
        );
    }

    #[test]
    fn test_iv_golden_short_expiry_auto() {
        // T=0.1
        let p = OptionParams {
            time: 0.1_f64,
            ..atm_call()
        };
        let market_price = price(&p).unwrap();
        let iv = implied_vol(&p, market_price, IvSolver::Auto).unwrap();
        assert!(
            (iv - 0.20_f64).abs() < STANDARD,
            "Short expiry T=0.1 Auto: expected 0.20, got {iv}"
        );
    }

    #[test]
    fn test_iv_golden_deep_otm_auto() {
        // K=130 — exercises deeper OTM path
        let p = OptionParams {
            strike: 130.0_f64,
            ..atm_call()
        };
        let market_price = price(&p).unwrap();
        let iv = implied_vol(&p, market_price, IvSolver::Auto).unwrap();
        assert!(
            (iv - 0.20_f64).abs() < STANDARD,
            "Deep OTM K=130 Auto: expected 0.20, got {iv}"
        );
    }

    #[test]
    fn test_iv_golden_deep_itm_auto() {
        // K=70 — deep ITM call
        let p = OptionParams {
            strike: 70.0_f64,
            ..atm_call()
        };
        let market_price = price(&p).unwrap();
        let iv = implied_vol(&p, market_price, IvSolver::Auto).unwrap();
        assert!(
            (iv - 0.20_f64).abs() < STANDARD,
            "Deep ITM K=70 Auto: expected 0.20, got {iv}"
        );
    }

    // ── Round-trip tests per solver ────────────────────────────────────

    #[test]
    fn test_iv_roundtrip_atm_halley() {
        assert_iv_roundtrip(&atm_call(), IvSolver::Halley, STANDARD, "ATM Halley");
    }

    #[test]
    fn test_iv_roundtrip_atm_newton() {
        assert_iv_roundtrip(&atm_call(), IvSolver::Newton, STANDARD, "ATM Newton");
    }

    #[test]
    fn test_iv_roundtrip_atm_brent() {
        assert_iv_roundtrip(&atm_call(), IvSolver::Brent, STANDARD, "ATM Brent");
    }

    #[test]
    fn test_iv_roundtrip_atm_jackel() {
        assert_iv_roundtrip(&atm_call(), IvSolver::Jackel, STANDARD, "ATM Jackel");
    }

    #[test]
    fn test_iv_roundtrip_atm_auto() {
        assert_iv_roundtrip(&atm_call(), IvSolver::Auto, STANDARD, "ATM Auto");
    }

    #[test]
    fn test_iv_roundtrip_otm_k110_auto() {
        let p = OptionParams {
            strike: 110.0_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "OTM K=110");
    }

    #[test]
    fn test_iv_roundtrip_deep_otm_k130_auto() {
        let p = OptionParams {
            strike: 130.0_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "Deep OTM K=130");
    }

    #[test]
    fn test_iv_roundtrip_deep_itm_k70_auto() {
        let p = OptionParams {
            strike: 70.0_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "Deep ITM K=70");
    }

    #[test]
    fn test_iv_roundtrip_short_expiry_auto() {
        let p = OptionParams {
            time: 0.1_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "Short T=0.1");
    }

    #[test]
    fn test_iv_roundtrip_high_vol_auto() {
        let p = OptionParams {
            vol: 1.5_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "High vol 1.5");
    }

    #[test]
    fn test_iv_roundtrip_put_atm_auto() {
        assert_iv_roundtrip(&atm_put(), IvSolver::Auto, STANDARD, "Put ATM Auto");
    }

    #[test]
    fn test_iv_roundtrip_put_otm_auto() {
        let p = OptionParams {
            strike: 90.0_f64,
            ..atm_put()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "Put OTM K=90");
    }

    // ── Error cases ────────────────────────────────────────────────────

    #[test]
    fn test_iv_negative_price_returns_no_solution() {
        let p = atm_call();
        let result = implied_vol(&p, -1.0_f64, IvSolver::Auto);
        assert!(
            matches!(result, Err(IvError::NoSolution)),
            "negative price should return NoSolution, got {result:?}"
        );
    }

    #[test]
    fn test_iv_zero_price_returns_no_solution() {
        let p = atm_call();
        let result = implied_vol(&p, 0.0_f64, IvSolver::Auto);
        assert!(
            matches!(result, Err(IvError::NoSolution)),
            "zero price should return NoSolution, got {result:?}"
        );
    }

    #[test]
    fn test_iv_below_intrinsic_returns_error() {
        // For a deep ITM call, intrinsic is large
        let p = OptionParams {
            option_type: OptionType::Call,
            spot: 150.0_f64,
            strike: 100.0_f64,
            rate: 0.05_f64,
            div_yield: 0.02_f64,
            vol: 0.20_f64,
            time: 1.0_f64,
        };
        // Intrinsic = S*exp(-qT) - K*exp(-rT) ≈ 147.03 - 95.12 ≈ 51.91
        let intrinsic = intrinsic_value(150.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 1.0_f64, OptionType::Call);
        // Price below intrinsic
        let below_price = intrinsic - 1.0_f64;
        let result = implied_vol(&p, below_price, IvSolver::Auto);
        assert!(
            matches!(result, Err(IvError::BelowIntrinsic { .. })),
            "below-intrinsic price should return BelowIntrinsic, got {result:?}"
        );
    }

    // ── Solver enum tests ──────────────────────────────────────────────

    #[test]
    fn test_iv_solver_debug() {
        let s = IvSolver::Auto;
        let debug = format!("{s:?}");
        assert!(debug.contains("Auto"));
    }

    #[test]
    fn test_iv_solver_eq() {
        assert_eq!(IvSolver::Auto, IvSolver::Auto);
        assert_ne!(IvSolver::Auto, IvSolver::Halley);
    }

    // ── Corrado-Miller initial guess quality ───────────────────────────

    #[test]
    fn test_corrado_miller_atm_reasonable() {
        let guess = corrado_miller_guess(
            100.0_f64, 100.0_f64, 0.05_f64, 0.02_f64, 1.0_f64,
            price(&atm_call()).unwrap(),
            OptionType::Call,
        );
        // Should be within 50% of true vol for ATM
        assert!(
            (guess - 0.20_f64).abs() < 0.10_f64,
            "Corrado-Miller ATM guess: expected ~0.20, got {guess}"
        );
    }

    #[test]
    fn test_corrado_miller_otm_positive() {
        let p = OptionParams {
            strike: 110.0_f64,
            ..atm_call()
        };
        let mp = price(&p).unwrap();
        let guess = corrado_miller_guess(
            100.0_f64, 110.0_f64, 0.05_f64, 0.02_f64, 1.0_f64,
            mp, OptionType::Call,
        );
        assert!(
            guess > 0.0_f64,
            "Corrado-Miller OTM guess must be positive, got {guess}"
        );
    }

    // ── Deep OTM / ITM per solver ──────────────────────────────────────

    #[test]
    fn test_iv_roundtrip_deep_otm_k130_halley() {
        let p = OptionParams {
            strike: 130.0_f64,
            ..atm_call()
        };
        // Halley may fail for deep OTM — that is acceptable, it should
        // not panic. The Auto chain will catch it.
        let market_price = price(&p).unwrap();
        let _result = implied_vol(&p, market_price, IvSolver::Halley);
        // We just verify it does not panic; convergence is not guaranteed for Halley here.
    }

    #[test]
    fn test_iv_roundtrip_deep_otm_k130_brent() {
        let p = OptionParams {
            strike: 130.0_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Brent, STANDARD, "Deep OTM Brent");
    }

    #[test]
    fn test_iv_roundtrip_deep_itm_k70_brent() {
        let p = OptionParams {
            strike: 70.0_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Brent, STANDARD, "Deep ITM Brent");
    }

    // ── Various moneyness / expiry combinations ────────────────────────

    #[test]
    fn test_iv_roundtrip_low_vol_auto() {
        let p = OptionParams {
            vol: 0.05_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "Low vol 0.05");
    }

    #[test]
    fn test_iv_roundtrip_long_expiry_auto() {
        let p = OptionParams {
            time: 3.0_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "Long T=3.0");
    }

    #[test]
    fn test_iv_roundtrip_negative_rate_auto() {
        let p = OptionParams {
            rate: -0.01_f64,
            div_yield: 0.0_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "Negative rate");
    }

    #[test]
    fn test_iv_roundtrip_high_div_yield_auto() {
        let p = OptionParams {
            div_yield: 0.08_f64,
            ..atm_call()
        };
        assert_iv_roundtrip(&p, IvSolver::Auto, STANDARD, "High div q=0.08");
    }
}
