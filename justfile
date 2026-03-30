# regit-blackscholes task runner

# Run all checks (format, lint, test, doc)
check: fmt lint test doc

# Format check
fmt:
    cargo fmt --check

# Lint — zero warnings
lint:
    cargo clippy -- -D warnings

# Run all tests
test:
    cargo test

# Run tests with SIMD feature
test-simd:
    cargo test --features simd

# Run benchmarks
bench:
    cargo bench --bench blackscholes

# Build documentation
doc:
    cargo doc --no-deps

# License audit
deny:
    cargo deny check

# Full CI pipeline
ci: fmt lint test test-simd doc deny

# Run property tests with extra cases
proptest:
    PROPTEST_CASES=5000 cargo test prop_

# Run Miri for undefined behaviour checks
miri:
    cargo +nightly miri test --lib
