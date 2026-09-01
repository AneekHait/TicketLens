# TicketLens — working notes

Offline PySide6 desktop app that clusters IT tickets with local embedding + LLM models.
Open source, Apache-2.0. See [README.md](README.md) for what it does and how a user
installs it, and [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor-facing version of
these notes; this file is only the things that aren't obvious from reading the code.

## Commands

```powershell
powershell -ExecutionPolicy Bypass -File check.ps1      # ruff + pytest — run before any build
powershell -ExecutionPolicy Bypass -File build_dist.ps1 # zip to the parent dir
.venv\Scripts\python.exe -m pytest tests -q             # tests alone
```

`check.ps1` is the gate; it prefers `.venv` so it checks what `run.bat` actually ships.
`.github/workflows/ci.yml` runs the same two commands (ruff, then pytest) on every push and
PR, so green locally should mean green on CI. Rebuild the zip after any source change — it
is easy to forget and ship a stale artifact.

Version lives in `src/__init__.py` only. `run.bat`, `build_dist.ps1` and the About dialog
all read it, and the build asserts the archived copy agrees.

## Architecture

`src/gui.py` is the **controller**: it owns `self.df`, the worker threads, and every signal
connection. `src/ui/` is presentation and must never import it — `theme.py` (palette,
stylesheet, widget factories) → `widgets.py` → `dialogs.py`, and `pages.py` off `theme`.
`tests/test_ui_layering.py` enforces the direction.

**Only `gui.py` and `wordcloud_studio.py` import PySide6.** Every other module in `src/` is
importable headless, which is why most of the suite needs no `QApplication`. Keep it that
way — `src/session.py` was written Qt-free on purpose.

Heavy imports are lazy (`get_pandas()`, `get_clusterer_class()`) for startup time. Don't
hoist them to module scope.

## Traps

Each of these has cost real debugging time here.

- **Never touch widgets from a worker thread.** Marshal with `self._post(fn)` (the
  `call_on_main` signal). `QTimer.singleShot` created on a thread with no event loop never
  fires — silently. `tests/test_thread_marshalling.py` guards this.

- **`except Exception as e:` — bind the message before deferring.** Python deletes `e` at
  block exit, so `self._post(lambda: show(str(e)))` raises `NameError` when it runs. Write
  `msg = str(e)` first. This silently killed four error dialogs.

- **Qt ignores a stylesheet `url()` it can't open** — no warning, no exception, it just
  draws nothing. A wrong asset path renders a tickless checkbox and a caret-less combo, and
  no test or linter sees it. `tests/test_theme_assets.py` exists because this shipped once.

- **`save_config` replaces the file.** `_gather_settings` must seed from
  `copy.deepcopy(self.config)` and overwrite only the blocks it owns, or sections with no UI
  are truncated out of `user_settings.json` on every RUN.

- **Progress must be monotonic.** `clustering.run()` reports `progress` at many points; a
  reordered stage easily rewinds the bar. `tests/test_run_progress.py` runs a stubbed
  pipeline and asserts non-decreasing.

- **JSON stringifies dict keys.** `cluster_data` is keyed by cluster ID (often
  `numpy.int64`), so anything that round-trips it through JSON must restore `int` keys —
  otherwise every `cluster_data[cid]` lookup misses and the failure is a silently empty
  result, not an error. See `_restore_int_keys` in `src/session.py`.

- **Keep `.ps1` files pure ASCII.** Windows PowerShell 5.1 reads them as ANSI without a BOM,
  so an em dash or arrow breaks parsing. `run.sh` is held to the same rule.

- **`run.sh` must be LF-only.** A CRLF shebang fails on Linux as
  `bad interpreter: /usr/bin/env bash^M` — which reads as a missing bash, not a line-ending
  bug. `.gitattributes` pins `*.sh eol=lf` and `tests/test_packaging.py` fails on CRLF,
  because editing it on Windows is exactly how this gets reintroduced.

- **Two launchers, one version.** `run.bat` (Windows) and `run.sh` (macOS/Linux) both parse
  `__version__` out of `src/__init__.py` and must stay behaviourally in step: venv repair,
  dependency install, one-time per-machine LLM health check, launch. `run.sh` inlines the
  health check because `repair_llm.ps1` is PowerShell-only, and it skips `wheels/` entirely
  (that wheel is `win_amd64`).

## Conventions

- **Cluster-label column lookup lives only in `src/column_utils.py`**
  (`resolve_label_column`, `CLUSTER_CATEGORY_COLS`, `CLUSTER_SUBCATEGORY_COLS`). It was
  duplicated across `impact_analysis.py` and `quality_audit.py` and caused two separate
  bugs. A test asserts it stays in one place.

- **All LLM calls go through `create_chat_completion`**, so the model supplies its own chat
  template. Never hardcode prompt markup — that mislabelled Gemma. A new curated model in
  `config.py:LLM_MODELS` must be an ungated GGUF with an embedded template.

- **Ruff is scoped to correctness** (`F`, `E9`, `B`, `ISC`); `E501` is ignored and `UP`/`C4`
  are deliberately unselected. Don't mass-reformat — see the comments in `ruff.toml`.
  Target is `py311` even though the venv is newer, so no 3.12+ syntax (notably: no
  backslash escapes inside f-string replacement fields).

- **`pytest.ini` pins `--basetemp=.pytest_tmp`** so the exit code is trustworthy; without it
  Windows symlink teardown returned 1 on a passing suite. Don't remove it.

- Tests are behavioural and named as sentences. When fixing a bug, add the test that fails
  before the fix — and check it isn't passing vacuously (a static scan of `gui.py` for code
  that has since moved will pass while asserting nothing).

## Never commit, never ship

`*.xlsx`, `*.csv`, `*.tsz` are gitignored because they hold **real support-ticket data**.
This is a public repo: a single careless `git add .` publishes a customer's tickets
irreversibly. `config/user_settings.json` and `config/custom_stopwords.txt` are per-user;
`build_dist.ps1` fails the build if either appears in the archive. Check `git status`
before a broad `git add`.

## Open source

Apache-2.0 (`LICENSE`), with third-party licenses catalogued in `NOTICE`. `build_dist.ps1`
lists `LICENSE`, `NOTICE` and `README.md` in `$required`, so the build fails rather than
shipping a zip that breaches the redistribution terms.

App identity lives in `src/ui/theme.py` (`APP_NAME`, `AUTHOR`, `PROJECT_URL`,
`AUTHOR_URL`, `ISSUES_URL`) and is rendered by the About dialog. There is deliberately
**no** integrity/tamper hash over those fields — the earlier build had one, and an
anti-tamper lock on the author line is meaningless in a repo anyone can fork and edit.

## Standing constraints

- **Do not remove `min_cluster_size` from the `disposition` section of `src/config.py`.**
  It looks redundant next to `clustering.min_cluster_size`; leave it.
