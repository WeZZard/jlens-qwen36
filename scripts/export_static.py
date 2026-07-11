#!/usr/bin/env python3
"""Export the J-Space visualizer as a fully static presentation build.

Produces a self-contained ``dist/`` deployable on any static host
(Cloudflare Pages, GitHub Pages, nginx, ``python -m http.server``):

- ``index.html``          web/index.html with JLENS_MODE baked to "presentation"
- ``meta.json``           model + lens info (queried from a running server if any)
- ``sessions/index.json`` manifest mirroring the /api/sessions list response
- ``sessions/<id>.<k>.gz`` CHUNKED, gzipped session payloads. Chunk 0 is
                          the conversation envelope (messages, markups,
                          settings, layers) plus the first snapshots —
                          enough for the client to render immediately —
                          and chunks 1..K-1 carry the remaining snapshots,
                          streamed in the background. Chunking keeps first
                          paint fast on slow links and every asset far
                          below Cloudflare Pages' 25 MiB cap.
- ``_headers``            Cloudflare Pages cache rules (harmless elsewhere)

Presentation builds never contact a server: the web page's data bridge
(see fetchMaybeGzippedJson in web/index.html) reads exactly this layout.

Sessions failing the integrity check (non-contiguous per-token snapshots,
i.e. quota-trimmed middles) are refused, so a lossy capture cannot be
published silently.

Usage:
  python3 scripts/export_static.py [--out dist] [--sessions id1,id2]
                                   [--include-autosave] [--server URL]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SESSIONS_DIR = REPO / "data" / "sessions"
WEB_INDEX = REPO / "web" / "index.html"

MODE_PLACEHOLDER = 'window.JLENS_MODE = "active"'

# Snapshots per chunk. ~80 token snapshots compress to roughly 200-400 KB —
# a 1-2s first paint even on slow links, while keeping the number of chunk
# requests modest for long sessions.
CHUNK_SNAPSHOTS = 80

CF_HEADERS = """\
/index.html
  Cache-Control: public, max-age=300

/meta.json
  Cache-Control: public, max-age=300

/sessions/index.json
  Cache-Control: public, max-age=300

/sessions/*.gz
  Cache-Control: public, max-age=31536000, immutable
"""


def snapshots_are_lossy(session: dict) -> bool:
    """Mirror of snapshotsAreLossy() in web/index.html: per generated
    message, per-token snapshots must be contiguous starting at token 0.
    A/B compares tag the inactive run's snapshots variant:'other' and each
    stream restarts snapshot ids, so group per (variant, message)."""
    by_msg: dict[str, list[int]] = {}
    for sn in session.get("snapshots") or []:
        if sn.get("token_idx", -1) == -1:
            continue
        key = f"{sn.get('variant') or ''}:{sn.get('msg_idx', 0)}"
        by_msg.setdefault(key, []).append(sn["token_idx"])
    for idxs in by_msg.values():
        idxs.sort()
        if idxs[0] != 0:
            return True
        if any(b - a > 1 for a, b in zip(idxs, idxs[1:])):
            return True
    return False


def fetch_meta(server: str) -> dict:
    try:
        with urllib.request.urlopen(f"{server}/api/model", timeout=3) as r:
            model = json.load(r)
        with urllib.request.urlopen(f"{server}/api/lens", timeout=3) as r:
            lens = json.load(r)
        model["mode"] = "presentation"
        print(f"meta: from live server {server}")
        return {"model": model, "lens": lens}
    except Exception as e:
        print(f"meta: no live server ({e}); writing placeholders")
        return {
            "model": {"model_id": "unknown", "mode": "presentation"},
            "lens": {"lens_loaded": False},
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dist", help="output directory (default: dist)")
    ap.add_argument("--sessions", default=None,
                    help="comma-separated session ids to export (default: all named sessions)")
    ap.add_argument("--include-autosave", action="store_true",
                    help="also export autosave-latest.json")
    ap.add_argument("--server", default="http://127.0.0.1:8765",
                    help="running server to query for meta.json (optional)")
    args = ap.parse_args()

    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    for stale in (out / "sessions").glob("*.gz"):
        stale.unlink()

    html = WEB_INDEX.read_text()
    if MODE_PLACEHOLDER not in html:
        print(f"ERROR: mode placeholder not found in {WEB_INDEX}", file=sys.stderr)
        return 1

    # Select sessions.
    if args.sessions:
        paths = [SESSIONS_DIR / s.strip() for s in args.sessions.split(",")]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"ERROR: session(s) not found: {[p.name for p in missing]}", file=sys.stderr)
            return 1
    else:
        paths = sorted(SESSIONS_DIR.glob("*.json"))
        if not args.include_autosave:
            paths = [p for p in paths if p.name != "autosave-latest.json"]

    manifest = []
    skipped = 0
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  SKIP {path.name}: unreadable ({e})")
            skipped += 1
            continue
        if snapshots_are_lossy(data):
            print(f"  SKIP {path.name}: INTEGRITY FAILURE (non-contiguous snapshots — "
                  f"quota-trimmed capture; re-save it from a fully recovered state)")
            skipped += 1
            continue
        snaps = data.get("snapshots") or []
        batches = [snaps[i:i + CHUNK_SNAPSHOTS]
                   for i in range(0, len(snaps), CHUNK_SNAPSHOTS)] or [[]]
        total_gz = 0
        compact = dict(ensure_ascii=False, separators=(",", ":"))
        for k, batch in enumerate(batches):
            if k == 0:
                payload = dict(data)
                payload["snapshots"] = batch
                payload["chunking"] = {"chunks": len(batches),
                                       "total_snapshots": len(snaps)}
            else:
                payload = {"snapshots": batch}
            gz = gzip.compress(json.dumps(payload, **compact).encode(), compresslevel=9)
            (out / "sessions" / f"{path.name}.{k}.gz").write_bytes(gz)
            total_gz += len(gz)
            if len(gz) > 25 * 1024 * 1024:
                print(f"  WARNING: {path.name}.{k} exceeds Cloudflare Pages' "
                      f"25 MiB asset cap; lower CHUNK_SNAPSHOTS")
        manifest.append({
            "id": path.name,
            "preview": data.get("preview") or "Untitled session",
            "created_at": data.get("created_at"),
            "saved_at": data.get("saved_at"),
            "message_count": len(data.get("messages") or []),
            "snapshot_count": len(snaps),
            "chunks": len(batches),
            "gz_bytes": total_gz,
        })
        print(f"  OK   {path.name}: {path.stat().st_size/1e6:.1f} MB -> "
              f"{total_gz/1e6:.1f} MB gzipped in {len(batches)} chunk(s)")

    manifest.sort(key=lambda m: m.get("saved_at") or "", reverse=True)
    meta = fetch_meta(args.server)
    # The manifest and meta are also written as files (fallback + tooling),
    # but the page INLINES them so the first screen renders with the HTML
    # itself — zero extra round trips on slow links.
    (out / "sessions" / "index.json").write_text(
        json.dumps({"sessions": manifest}, ensure_ascii=False, indent=2)
    )
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    (out / "_headers").write_text(CF_HEADERS)

    def _inline_json(value) -> str:
        # "</" would terminate the surrounding <script> tag early.
        return json.dumps(value, ensure_ascii=False,
                          separators=(",", ":")).replace("</", "<\\/")

    baked = (
        'window.JLENS_MODE = "presentation";\n'
        f"  window.JLENS_SESSIONS = {_inline_json(manifest)};\n"
        f"  window.JLENS_META = {_inline_json(meta)};"
    )
    (out / "index.html").write_text(
        html.replace(MODE_PLACEHOLDER + ";", baked, 1)
    )

    print(f"\nexported {len(manifest)} session(s) ({skipped} skipped) -> {out}")
    if not manifest:
        print("ERROR: nothing exported", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
