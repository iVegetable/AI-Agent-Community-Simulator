# Security Policy

## Supported Versions

This project is maintained on a best-effort basis. Security fixes are applied to the latest `main` branch.

## Reporting a Vulnerability

Please do **not** open a public issue for security vulnerabilities.

Report privately via:

- GitHub Security Advisories (preferred): use the "Report a vulnerability" button in this repository
- Or email the maintainer directly if available in the repository profile

When reporting, include:

- A clear description of the issue
- Steps to reproduce
- Impact assessment
- Any proof-of-concept details

We will acknowledge receipt as soon as possible and provide updates on triage and remediation.

## Secrets and Keys

- Never commit `.env` files or API keys.
- Rotate any leaked credentials immediately (for example OpenAI keys).
- Use `backend/.env.example` as the template for local configuration.
