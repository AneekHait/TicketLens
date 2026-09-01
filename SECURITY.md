# Security Policy

## Supported versions

TicketLens is a single-developer project. Only the latest release on `main`
receives security fixes.

## Reporting a vulnerability

Please **do not open a public issue** for a security vulnerability.

Use GitHub's private reporting instead:
[Report a vulnerability](https://github.com/AneekHait/TicketLens/security/advisories/new).

Include what the issue is, how to reproduce it, and what an attacker could
achieve. I will acknowledge within a few days and aim to ship a fix or a
mitigation before any public disclosure.

## Scope notes

TicketLens is an offline desktop application. It has no server, no accounts and
no network listener. The realistic threat surface is:

- **Untrusted input files.** Excel workbooks and `.tsz` session files are parsed
  on load. A malicious workbook or session file that achieves code execution is
  in scope.
- **Model files.** `.gguf` models are loaded by `llama.cpp`. Only load models from
  sources you trust; a malicious GGUF is a real risk, and issues there belong
  upstream with `llama.cpp` rather than here.
- **Downloads.** Models are fetched from Hugging Face over TLS on first use. A
  flaw in how TicketLens resolves or verifies those downloads is in scope.

Out of scope: vulnerabilities in third-party dependencies (report them upstream,
though a heads-up here is welcome so the pin can be bumped), and anything that
requires an attacker to already have write access to the install directory.

## Privacy

TicketLens sends no telemetry and makes no network calls other than model
downloads. Ticket data never leaves the machine. If you observe otherwise, treat
it as a security bug and report it as above.
