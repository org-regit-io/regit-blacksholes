# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in `regit-blackscholes`, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email **security@regit.io** with a description of the vulnerability
3. Include steps to reproduce if possible
4. We will acknowledge receipt within 48 hours and provide a timeline for a fix

## Scope

This crate performs mathematical computation only — it does not handle network I/O, file I/O, user authentication, or any form of external communication. The primary security concern is **numerical correctness**: an error in pricing or Greeks computation could lead to incorrect financial decisions.

If you find a numerical accuracy issue that falls outside the documented tolerance bounds, please report it using the same process above.
