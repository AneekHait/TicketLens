"""Save and reload a completed clustering run.



Before this, a finished run lived only in memory. The one durable output was an Excel

export, and reloading that export ran ``_reset_derived_state``, which drops

``cluster_data`` -- so six of the ten analysis tabs refused to open and every

LLM-derived result had to be paid for again. ``config.py``'s module docstring had

claimed "session persistence" for a while; this is it.



Format: a zip (``.tsz``) holding



    manifest.json   format version, app version, timestamp, source file, sheet

    settings.json   the settings the run used, for provenance

    tickets.csv     the labelled frame

    state.json      cluster_data and the LLM-derived analysis results



Why a zip of text and not the obvious alternatives:



* **pickle** is version-brittle and executes arbitrary code on load. These files are

  meant to be handed to a colleague, which makes that a poor trade.

* **parquet** needs ``pyarrow``, which is only an *optional extra* of pandas 3.x and is

  absent from ``requirements.lock``. It is present in some dev environments purely as a

  transitive dependency, so relying on it would break a clean install.



CSV loses some dtype fidelity. That is acceptable because the app already round-trips

this frame through Excel, and the consumers re-coerce (``pd.to_datetime(...,

errors="coerce")``, ``pd.to_numeric``) rather than trusting dtypes.



What is deliberately *not* stored, and why:



* ``clusterer`` -- holds a live llama-cpp handle and the fitted BERTopic model. On

  reload the LLM-backed features (KBA, SOP, Category Audit, the word cloud's "cluster

  topics" source) correctly report that no model is loaded; they already guard for it.

* ``topic_embedding`` inside ``cluster_data`` -- a 768-float vector per cluster that

  nothing outside ``clustering.py`` reads.

* The pandas-derived analysis results (``problem_clusters``, ``kpi``,

  ``business_process``, ``impact_figs``). They hold DataFrames and Plotly figures, and

  they are deterministic, LLM-free and quick to rebuild -- so they are recomputed on

  demand rather than persisted. Only results that cost LLM tokens are stored.

"""



import io

import json

import os

import zipfile

from datetime import datetime, timezone



SESSION_FORMAT = 1

# Kept as .tsz through the TicketScope -> TicketLens rename. The letters are
# not load-bearing, and the extension is named in two customer-data guards
# (.gitignore and build_dist.ps1's $excludeExt) that must not silently drift.
SESSION_EXT = ".tsz"



_MANIFEST = "manifest.json"

_SETTINGS = "settings.json"

_TICKETS = "tickets.csv"

_STATE = "state.json"



# Analysis results worth persisting: each was produced by the LLM, or is a plain summary

# dict. Anything not listed is recomputed on demand (see the module note).

PERSISTED_ANALYSIS_KEYS = (

    "disposition",

    "disposition_summary",

    "audit_summary",

    "main_theme",

)



# Results that are dropped on save and rebuilt by pressing the feature's button again.

RECOMPUTED_ANALYSIS_KEYS = (

    "problem_clusters",

    "kpi",

    "business_process",

    "impact_figs",

)





class SessionError(Exception):

    """Raised when a session file is unreadable or from an unsupported version."""





def _jsonify(obj):

    """Convert numpy/pandas scalars to plain Python; drop anything unserialisable.



    Returns ``(value, skipped)``, where ``skipped`` lists the dotted paths that were

    dropped -- so the caller can say what did not survive instead of writing a file with

    silent holes in it.

    """

    skipped = []



    def walk(o, path):

        # numpy scalars (and anything else exposing .item()) -> Python scalar

        if hasattr(o, "item") and not hasattr(o, "__len__"):

            try:

                return o.item()

            except Exception:

                pass

        if o is None or isinstance(o, (bool, int, float, str)):

            return o

        if isinstance(o, dict):

            out = {}

            for k, v in o.items():

                kk = k.item() if hasattr(k, "item") else k

                out[str(kk)] = walk(v, f"{path}.{kk}")

            return out

        if isinstance(o, (list, tuple, set)):

            return [walk(v, f"{path}[{i}]") for i, v in enumerate(o)]

        # numpy arrays / pandas Series expose tolist(); DataFrames do not.

        if hasattr(o, "tolist") and not hasattr(o, "columns"):

            try:

                return walk(o.tolist(), path)

            except Exception:

                pass

        skipped.append(path)

        return None



    return walk(obj, "root"), skipped





def _restore_int_keys(d):

    """Turn cluster-ID keys back into ints (including the -1 noise bucket).



    JSON forces dict keys to strings. Without this every ``cluster_data[cluster_id]``

    lookup would silently miss.

    """

    if not isinstance(d, dict):

        return d

    out = {}

    for k, v in d.items():

        try:

            out[int(k)] = v

        except (TypeError, ValueError):

            out[k] = v

    return out





def save_session(path, *, df, cluster_data=None, analysis_results=None,

                 kba_articles=None, sop_documents=None, selected_text_cols=None,

                 settings=None, source_file=None, sheet_name=None,

                 app_version="unknown"):

    """Write a session zip. Returns the list of dropped field paths (usually empty)."""

    if df is None:

        raise SessionError(

            "There is nothing to save yet - load data and run clustering first.")



    cluster_data = cluster_data or {}

    analysis_results = analysis_results or {}



    # Drop the unread per-cluster embedding before serialising (see module note).

    slim_clusters = {}

    for cid, cdata in cluster_data.items():

        slim_clusters[cid] = {k: v for k, v in dict(cdata).items()

                              if k != "topic_embedding"}



    keep = {k: analysis_results[k] for k in PERSISTED_ANALYSIS_KEYS

            if k in analysis_results}



    state, skipped = _jsonify({

        "cluster_data": slim_clusters,

        "analysis_results": keep,

        "kba_articles": kba_articles or [],

        "sop_documents": sop_documents or [],

        "selected_text_cols": list(selected_text_cols or []),

    })



    manifest = {

        "format": SESSION_FORMAT,

        "app_version": app_version,

        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),

        "source_file": os.path.basename(source_file) if source_file else None,

        "sheet_name": sheet_name,

        "rows": int(len(df)),

        "columns": [str(c) for c in df.columns],

        "clusters": len(slim_clusters),

        "dropped_fields": skipped,

        "recompute_on_load": list(RECOMPUTED_ANALYSIS_KEYS),

    }



    settings_blob, _ = _jsonify(settings or {})



    csv_buf = io.StringIO()

    df.to_csv(csv_buf, index=False)



    # Write beside the target and rename, so an interrupted save never replaces a good

    # session file with a truncated one.

    tmp = f"{path}.part"

    try:

        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:

            z.writestr(_MANIFEST, json.dumps(manifest, indent=2))

            z.writestr(_SETTINGS, json.dumps(settings_blob, indent=2))

            z.writestr(_STATE, json.dumps(state, indent=2))

            z.writestr(_TICKETS, csv_buf.getvalue())

        os.replace(tmp, path)

    except Exception:

        if os.path.exists(tmp):

            try:

                os.remove(tmp)

            except OSError:

                pass

        raise

    return skipped





def read_manifest(path):

    """Read just the manifest, without decoding the frame."""

    try:

        with zipfile.ZipFile(path) as z:

            return json.loads(z.read(_MANIFEST).decode("utf-8"))

    except (zipfile.BadZipFile, KeyError) as e:

        raise SessionError(

            f"{os.path.basename(path)} is not a TicketLens session file.") from e

    except (OSError, json.JSONDecodeError) as e:

        raise SessionError(f"Could not read {os.path.basename(path)}: {e}") from e





def load_session(path):

    """Load a session zip into a plain dict of restored state."""

    manifest = read_manifest(path)



    fmt = manifest.get("format")

    if not isinstance(fmt, int):

        raise SessionError(

            "This session file has no format version and cannot be read.")

    if fmt > SESSION_FORMAT:

        raise SessionError(

            f"This session was written by a newer version of TicketLens "

            f"(format {fmt}; this build reads up to {SESSION_FORMAT}). "

            f"Update the app to open it.")



    try:

        with zipfile.ZipFile(path) as z:

            state = json.loads(z.read(_STATE).decode("utf-8"))

            settings = json.loads(z.read(_SETTINGS).decode("utf-8"))

            csv_bytes = z.read(_TICKETS)

    except (KeyError, zipfile.BadZipFile) as e:

        raise SessionError(

            f"{os.path.basename(path)} is missing part of its contents "

            f"and cannot be opened.") from e

    except (OSError, json.JSONDecodeError) as e:

        raise SessionError(f"Could not read {os.path.basename(path)}: {e}") from e



    import pandas as pd

    df = pd.read_csv(io.BytesIO(csv_bytes))



    cluster_data = _restore_int_keys(state.get("cluster_data") or {})



    analysis = dict(state.get("analysis_results") or {})

    # audit_summary["clusters"] is cluster-keyed too, and the per-cluster quality table

    # reads it back by ID.

    if isinstance(analysis.get("audit_summary"), dict):

        aud = dict(analysis["audit_summary"])

        if isinstance(aud.get("clusters"), dict):

            aud["clusters"] = _restore_int_keys(aud["clusters"])

        analysis["audit_summary"] = aud



    return {

        "manifest": manifest,

        "settings": settings,

        "df": df,

        "cluster_data": cluster_data,

        "analysis_results": analysis,

        "kba_articles": state.get("kba_articles") or [],

        "sop_documents": state.get("sop_documents") or [],

        "selected_text_cols": state.get("selected_text_cols") or [],

    }

