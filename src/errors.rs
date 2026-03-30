// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Typed error enums for pricing and implied volatility operations.
//!
//! All failure paths return typed `Result` — no `panic!()`, no `unwrap()`,
//! no string errors. Each variant carries enough context for the caller
//! to decide how to recover.

use core::fmt;

/// Error returned by pricing functions when input validation fails
/// or the option is at a degenerate boundary.
///
/// # Variants
///
/// All variants indicate a precondition violation except `IntrinsicOnly`,
/// which signals a valid but degenerate edge case (`T == 0`).
///
/// # Examples
///
/// ```
/// use regit_blackscholes::errors::PricingError;
///
/// let err = PricingError::NegativeSpot;
/// assert_eq!(format!("{err}"), "spot price must be non-negative");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PricingError {
    /// Spot price `S` is negative.
    NegativeSpot,
    /// Strike price `K` is negative.
    NegativeStrike,
    /// Time to expiry `T` is negative.
    NegativeTime,
    /// Volatility `sigma` is negative.
    NegativeVolatility,
    /// Time to expiry is zero — the option value equals the discounted
    /// intrinsic value. The `intrinsic` field carries that value so the
    /// caller can use it without re-computing.
    IntrinsicOnly {
        /// The discounted intrinsic value of the option at expiry.
        intrinsic: f64,
    },
}

impl fmt::Display for PricingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeSpot => write!(f, "spot price must be non-negative"),
            Self::NegativeStrike => write!(f, "strike price must be non-negative"),
            Self::NegativeTime => write!(f, "time to expiry must be non-negative"),
            Self::NegativeVolatility => write!(f, "volatility must be non-negative"),
            Self::IntrinsicOnly { intrinsic } => {
                write!(f, "option at expiry: intrinsic value = {intrinsic}")
            }
        }
    }
}

impl std::error::Error for PricingError {}

/// Error returned by implied volatility solvers when convergence fails
/// or the market price is inconsistent with the model.
///
/// # Variants
///
/// Each variant carries diagnostic context so the caller can decide
/// whether to retry with a different solver or report the failure.
///
/// # Examples
///
/// ```
/// use regit_blackscholes::errors::IvError;
///
/// let err = IvError::NoSolution;
/// assert_eq!(format!("{err}"), "no implied volatility solution exists");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IvError {
    /// No implied volatility solution exists for the given market price.
    NoSolution,
    /// Market price is below the intrinsic value — no valid vol can produce it.
    BelowIntrinsic {
        /// The intrinsic value that the market price falls below.
        intrinsic: f64,
    },
    /// Solver reached maximum iteration count without converging.
    MaxIterationsReached {
        /// The last volatility estimate before iteration stopped.
        last_vol: f64,
        /// The residual (price error) at the last iteration.
        residual: f64,
    },
    /// Vega is near zero — the solver cannot make progress because
    /// the price surface is flat with respect to volatility.
    NearZeroVega,
    /// The implied volatility solution exceeds the search bounds `[1e-8, 100.0]`.
    BoundsExceeded {
        /// The volatility value that exceeded the bounds.
        vol: f64,
    },
}

impl fmt::Display for IvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSolution => write!(f, "no implied volatility solution exists"),
            Self::BelowIntrinsic { intrinsic } => {
                write!(
                    f,
                    "market price is below intrinsic value ({intrinsic})"
                )
            }
            Self::MaxIterationsReached { last_vol, residual } => {
                write!(
                    f,
                    "IV solver did not converge: last_vol = {last_vol}, residual = {residual}"
                )
            }
            Self::NearZeroVega => write!(f, "vega is near zero — solver cannot make progress"),
            Self::BoundsExceeded { vol } => {
                write!(f, "implied volatility {vol} exceeds search bounds")
            }
        }
    }
}

impl std::error::Error for IvError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pricing_error_display_negative_spot() {
        let err = PricingError::NegativeSpot;
        assert_eq!(format!("{err}"), "spot price must be non-negative");
    }

    #[test]
    fn test_pricing_error_display_negative_strike() {
        let err = PricingError::NegativeStrike;
        assert_eq!(format!("{err}"), "strike price must be non-negative");
    }

    #[test]
    fn test_pricing_error_display_negative_time() {
        let err = PricingError::NegativeTime;
        assert_eq!(format!("{err}"), "time to expiry must be non-negative");
    }

    #[test]
    fn test_pricing_error_display_negative_volatility() {
        let err = PricingError::NegativeVolatility;
        assert_eq!(format!("{err}"), "volatility must be non-negative");
    }

    #[test]
    fn test_pricing_error_display_intrinsic_only() {
        let err = PricingError::IntrinsicOnly {
            intrinsic: 5.25_f64,
        };
        assert_eq!(
            format!("{err}"),
            "option at expiry: intrinsic value = 5.25"
        );
    }

    #[test]
    fn test_pricing_error_is_error_trait() {
        let err: &dyn std::error::Error = &PricingError::NegativeSpot;
        assert!(err.source().is_none());
    }

    #[test]
    fn test_pricing_error_clone_copy() {
        let err = PricingError::NegativeSpot;
        let err2 = err;
        assert_eq!(err, err2);
    }

    #[test]
    fn test_pricing_error_debug() {
        let err = PricingError::NegativeSpot;
        let debug = format!("{err:?}");
        assert!(debug.contains("NegativeSpot"));
    }

    #[test]
    fn test_iv_error_display_no_solution() {
        let err = IvError::NoSolution;
        assert_eq!(
            format!("{err}"),
            "no implied volatility solution exists"
        );
    }

    #[test]
    fn test_iv_error_display_below_intrinsic() {
        let err = IvError::BelowIntrinsic {
            intrinsic: 10.0_f64,
        };
        assert_eq!(
            format!("{err}"),
            "market price is below intrinsic value (10)"
        );
    }

    #[test]
    fn test_iv_error_display_max_iterations() {
        let err = IvError::MaxIterationsReached {
            last_vol: 0.25_f64,
            residual: 0.001_f64,
        };
        let msg = format!("{err}");
        assert!(msg.contains("last_vol = 0.25"));
        assert!(msg.contains("residual = 0.001"));
    }

    #[test]
    fn test_iv_error_display_near_zero_vega() {
        let err = IvError::NearZeroVega;
        let msg = format!("{err}");
        assert!(msg.contains("vega is near zero"));
    }

    #[test]
    fn test_iv_error_display_bounds_exceeded() {
        let err = IvError::BoundsExceeded { vol: 150.0_f64 };
        let msg = format!("{err}");
        assert!(msg.contains("150"));
        assert!(msg.contains("exceeds search bounds"));
    }

    #[test]
    fn test_iv_error_is_error_trait() {
        let err: &dyn std::error::Error = &IvError::NoSolution;
        assert!(err.source().is_none());
    }

    #[test]
    fn test_iv_error_clone_copy() {
        let err = IvError::NearZeroVega;
        let err2 = err;
        assert_eq!(err, err2);
    }

    #[test]
    fn test_iv_error_debug() {
        let err = IvError::BoundsExceeded { vol: 200.0_f64 };
        let debug = format!("{err:?}");
        assert!(debug.contains("BoundsExceeded"));
    }

    #[test]
    fn test_pricing_error_eq() {
        assert_eq!(PricingError::NegativeSpot, PricingError::NegativeSpot);
        assert_ne!(PricingError::NegativeSpot, PricingError::NegativeStrike);
    }

    #[test]
    fn test_iv_error_eq() {
        assert_eq!(IvError::NoSolution, IvError::NoSolution);
        assert_ne!(IvError::NoSolution, IvError::NearZeroVega);
    }

    #[test]
    fn test_pricing_error_intrinsic_only_zero() {
        let err = PricingError::IntrinsicOnly {
            intrinsic: 0.0_f64,
        };
        if let PricingError::IntrinsicOnly { intrinsic } = err {
            assert!((intrinsic - 0.0_f64).abs() < 1e-15_f64);
        }
    }

    #[test]
    fn test_iv_error_max_iterations_fields() {
        let err = IvError::MaxIterationsReached {
            last_vol: 0.3_f64,
            residual: 1e-8_f64,
        };
        if let IvError::MaxIterationsReached { last_vol, residual } = err {
            assert!((last_vol - 0.3_f64).abs() < 1e-15_f64);
            assert!((residual - 1e-8_f64).abs() < 1e-20_f64);
        }
    }
}
