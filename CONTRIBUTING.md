# Contributing to TicketLens

Thanks for taking the time. TicketLens is a small, focused desktop app, and the
bar for a change is simple: it should make the tool more useful without making it
less private or harder to install.

## Ground rules

- **Never commit ticket data.** `*.xlsx`, `*.csv`, `*.docx` and `*.tsz` are
  gitignored because they hold real support tickets. Check `git status` before a
  broad `git add`. A PR containing a spreadsheet will be closed, not merged.
- **Offline stays offline.** The app must keep working with no network beyond the
  one-time Hugging Face model download. No telemetry, no analytics, no "phone
  home" — not even opt-in.
- **No new heavyweight dependencies** without discussing it in an issue first.
  Install size and cold-start time are features here.

## Getting set up

```powershell
git clone https://github.com/AneekHait/TicketLens.git
cd TicketLens
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu `
  --only-binary llama-cpp-python
```

On Windows you can just double-click `run.bat`, and on macOS/Linux run `./run.sh` --
either does all of the above and launches the app.

## Before you open a PR

```powershell
powershell -ExecutionPolicy Bypass -File check.ps1   # ruff + pytest
```

On macOS/Linux there is no `check.ps1` equivalent yet; run the two commands directly:

```bash
.venv/bin/python -m ruff check src tests main.py --output-format concise
.venv/bin/python -m pytest tests -q
```

`check.ps1` is the gate. It prefers `.venv`, so it checks the same interpreter
`run.bat` ships. CI runs the same two commands on every push and pull request.

## Code layout

`src/gui.py` is the **controller**: it owns the dataframe, the worker threads and
every signal connection. `src/ui/` is presentation and must never import it —
the dependency direction is `theme.py` → `widgets.py` → `dialogs.py`, with
`pages.py` off `theme.py`. `tests/test_ui_layering.py` enforces this.

Only `gui.py` and `wordcloud_studio.py` import PySide6. Everything else in `src/`
is importable headless, which is why most of the suite needs no `QApplication`.
Please keep it that way.

## Traps worth knowing

Each of these has cost real debugging time:

- **Never touch widgets from a worker thread.** Marshal with `self._post(fn)`.
  `QTimer.singleShot` on a thread with no event loop never fires, silently.
- **`except Exception as e:` — bind the message before deferring.** Python deletes
  `e` at block exit, so `self._post(lambda: show(str(e)))` raises `NameError` when
  it runs. Write `msg = str(e)` first.
- **Qt silently ignores a stylesheet `url()` it cannot open.** A wrong asset path
  renders a tickless checkbox and no linter sees it. See
  `tests/test_theme_assets.py`.
- **`save_config` replaces the file.** `_gather_settings` must seed from a
  `deepcopy` of the current config, or sections with no UI get truncated away.
- **JSON stringifies dict keys.** `cluster_data` is keyed by cluster ID (often
  `numpy.int64`); anything round-tripping it through JSON must restore `int` keys.
  See `_restore_int_keys` in `src/session.py`.
- **Keep `.ps1`, `.bat` and `.sh` files pure ASCII.** Windows PowerShell 5.1 reads
  `.ps1`/`.bat` as ANSI without a BOM, so an em dash breaks parsing.
- **`run.sh` must stay LF-only.** A CRLF shebang makes Linux report
  `bad interpreter: /usr/bin/env bash^M`, which looks like a missing bash rather
  than a line-ending problem. `.gitattributes` pins it and a test guards it.

## Style

Ruff is deliberately scoped to correctness (`F`, `E9`, `B`, `ISC`). `E501` is
ignored and `UP`/`C4` are unselected on purpose — see the comments in
`ruff.toml`. **Please don't mass-reformat.** A diff that reformats a file you also
changed is very hard to review.

Target is `py311`, so no 3.12+ syntax (notably: no backslash escapes inside
f-string replacement fields).

## Tests

Tests are behavioural and named as sentences. When you fix a bug, add the test
that fails *before* the fix — and check it isn't passing vacuously.

## Commits and PRs

Write a commit message that says what changed and why. Keep one logical change per
PR. If it fixes an issue, reference it.

## Reporting bugs

Open a [GitHub issue](https://github.com/AneekHait/TicketLens/issues) with your
OS, Python version, TicketLens version (Help → About), and the relevant lines from
`logs/`. **Redact ticket text before pasting logs.**

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, the same terms that cover the project.
