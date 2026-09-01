## What this changes

<!-- One or two sentences. What is different after this PR? -->

## Why

<!-- The problem it solves. Link the issue if there is one: Fixes #123 -->

## How it was verified

<!-- Which of these did you actually run? Paste output if a test is the point of the PR. -->

- [ ] `powershell -ExecutionPolicy Bypass -File check.ps1` passes (ruff + pytest)
- [ ] Added or updated a test that fails without this change
- [ ] Launched the app and exercised the affected screen

## Checklist

- [ ] No ticket data, spreadsheets, `.tsz` sessions or `config/user_settings.json` in the diff
- [ ] No new runtime dependency (or it was agreed in an issue first)
- [ ] No unrelated reformatting
- [ ] `.ps1` / `.bat` files touched are still pure ASCII
