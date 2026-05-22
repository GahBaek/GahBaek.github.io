#!/usr/bin/env python3
"""
Conference Deadline Crawler
─────────────────────────────────────────────────────
Sources (priority order — later sources fill gaps only):
  1. conferences.manual.yml  ← repo root, user-maintained (highest priority)
  2. sec-deadlines GitHub YAML
  3. se-deadlines  GitHub YAML
  4. ai-deadlines  GitHub YAML

Output: data.json in repo root
─────────────────────────────────────────────────────
"""

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
MANUAL_FILE = ROOT / "conferences.manual.yml"
OUT_FILE    = ROOT / "data.json"

# ── Remote YAML sources ───────────────────────────────────────────────────────
REMOTE_SOURCES = [
    {
        "name": "sec-deadlines",
        "url":  "https://raw.githubusercontent.com/sec-deadlines/sec-deadlines.github.io/master/_data/conferences.yml",
        "fmt":  "sec",   # fields: name / description / deadline / place / date / link
    },
    {
        "name": "se-deadlines",
        "url":  "https://raw.githubusercontent.com/se-deadlines/se-deadlines.github.io/master/_data/conferences.yml",
        "fmt":  "sec",
    },
    {
        "name": "ai-deadlines",
        "url":  "https://raw.githubusercontent.com/abhshkdz/ai-deadlines/gh-pages/_data/conferences.yml",
        "fmt":  "ai",    # fields: title / full_name / deadline / abstract_deadline / place / date / link
    },
]

# ── Whitelist: which conferences to keep from remote sources ──────────────────
# Regex matched against lowercase name/title; first match wins.
# Only remote entries need whitelisting — manual entries are always included.
WHITELIST = [
    # SE1
    (r"^icse\b(?!.*(?:seip|seis|nier|t&tb|jf|ict|ds |src|nfs|ip |ivr))", "SE1"),
    (r"^fse\b(?!.*(?:ip |ivr|jf ))",  "SE1"),
    (r"^esec/fse\b",                   "SE1"),
    # SE2
    (r"^ase\b(?!.*(?:nier|td |jf |is |src|ds ))", "SE2"),
    # AI1
    (r"^ijcai\b",    "AI1"),
    (r"^aaai\b(?!\s+\d{4}\s+(?:spring|fall|winter|summer))", "AI1"),
    (r"^www\b",      "AI1"),
    # Sec1
    (r"^s&p\b",                              "Sec1"),
    (r"^ieee s&p\b",                         "Sec1"),
    (r"^ccs\b(?!.*(?:workshop|ccsw|\sw\b))", "Sec1"),
    (r"^usenix sec",                         "Sec1"),
    (r"^ndss\b",                             "Sec1"),
    # Sec2
    (r"^acsac\b",    "Sec2"),
    (r"^raid\b",     "Sec2"),
    (r"^asiaccs\b",  "Sec2"),
    (r"^asia ccs\b", "Sec2"),
    (r"^esorics\b",  "Sec2"),
    (r"^euro s&p\b", "Sec2"),
    (r"^eurosp\b",   "Sec2"),
    (r"^dsn\b",      "Sec2"),
    # Sec3
    (r"^codaspy\b",    "Sec3"),
    (r"^dimva\b",      "Sec3"),
    (r"^securecomm\b", "Sec3"),
    # Sec4
    (r"^sac\b(?!.*(?:mat|sec|net|ped|svt|dac|dlt|dsp))", "Sec4"),
    (r"^ifip.?sec\b",  "Sec4"),
]
_COMPILED = [(re.compile(p, re.I), t) for p, t in WHITELIST]

TIER_ORDER = {t: i for i, t in enumerate(
    ["Sys1", "SE1", "SE2", "AI1", "Sec1", "Sec2", "Sec3", "Sec4"]
)}

# ── Name normalisation for dedup ──────────────────────────────────────────────
_ALIASES = {
    "ifip sec":        "IFIP-SEC",
    "ifip-sec":        "IFIP-SEC",
    "asia ccs":        "ASIACCS",
    "asiaccs":         "ASIACCS",
    "s&p (oakland)":   "S&P",
    "ieee s&p":        "S&P",
    "usenix security": "USENIX Security",
    "esec/fse":        "FSE",
    "euro s&p":        "Euro S&P",
    "eurosp":          "Euro S&P",
}

def _canon(name: str) -> str:
    n = name.lower().strip()
    for alias, canon in _ALIASES.items():
        if n.startswith(alias):
            return canon.lower()
    return re.split(r"[\s'(]", n)[0]   # first word

def _dedup_key(name: str, year: str) -> str:
    return f"{_canon(name)}|{year}"

def _match_tier(name: str) -> str | None:
    n = name.lower().strip()
    for pat, tier in _COMPILED:
        if pat.match(n):
            return tier
    return None

# ── Minimal YAML parser (no external deps) ───────────────────────────────────
def _parse_yaml(text: str) -> list[dict]:
    """
    Parse the sec-deadlines / ai-deadlines YAML list format.
    Each entry starts with a top-level '- key:' line.
    Handles:
      - scalar values
      - multi-line block lists (  - item)
      - inline lists [a, b, c]
    """
    # Split on top-level list items
    raw_blocks = re.split(r'\n(?=- \w)', "\n" + text)
    entries = []
    for block in raw_blocks:
        block = block.strip()
        if not block or not block.startswith("-"):
            continue
        entry: dict = {}
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m = re.match(r'^[-\s]\s*([a-zA-Z_]+):\s*(.*)', line)
            if not m:
                i += 1
                continue
            key = m.group(1).strip()
            val = m.group(2).strip()

            # Inline list: [a, b, c]
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                entry[key] = [x.strip().strip('"').strip("'")
                               for x in inner.split(",") if x.strip()] if inner else []
                i += 1
                continue

            # Block list: next lines start with 2+ spaces + "- "
            sub, j = [], i + 1
            while j < len(lines) and re.match(r'^\s{2,}-\s', lines[j]):
                item = re.sub(r'^\s+-\s*', '', lines[j]).strip().strip('"').strip("'")
                sub.append(item)
                j += 1
            if sub:
                entry[key] = sub
                i = j
                continue

            # Scalar
            entry[key] = val.strip('"').strip("'")
            i += 1

        if entry:
            entries.append(entry)
    return entries

# ── Date helpers ──────────────────────────────────────────────────────────────
def _extract_date(s) -> str | None:
    if not s:
        return None
    cur = datetime.now(timezone.utc).year
    s = str(s).replace("%Y", str(cur)).replace("%y", str(cur + 1))
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None

def _norm_deadlines(raw) -> list[str]:
    if not raw:
        return []
    items = [raw] if isinstance(raw, str) else list(raw)
    return [d for item in items if (d := _extract_date(str(item)))]

# ── Convert raw YAML entry → normalised conf dict ─────────────────────────────
def _from_manual(entry: dict) -> dict | None:
    """Parse an entry from conferences.manual.yml (same format as sec-deadlines)."""
    name = str(entry.get("name", "")).strip()
    tier = str(entry.get("tier", "")).strip()
    if not name or not tier:
        return None
    year = str(entry.get("year", "")).strip()

    raw_dl  = entry.get("deadline", [])
    raw_abs = entry.get("abstract_deadline", [])
    deadlines = _norm_deadlines(raw_dl)
    abstracts = _norm_deadlines(raw_abs)

    return {
        "name":       f"{name} '{year[-2:]}" if year else name,
        "full":       str(entry.get("description", "") or "").strip(),
        "tier":       tier,
        "year":       year,
        "deadlines":  deadlines,
        "abstracts":  abstracts,   # per-cycle abstract deadlines
        "venue":      str(entry.get("place", "") or "").strip(),
        "date":       str(entry.get("date", "") or "").strip(),
        "url":        str(entry.get("link", "") or "").strip(),
        "note":       str(entry.get("note", "") or "").strip(),
        "_source":    "manual",
    }

def _from_sec(entry: dict) -> dict | None:
    name = str(entry.get("name", "")).strip()
    tier = _match_tier(name)
    if not tier:
        return None
    year = str(entry.get("year", "")).strip()
    note = str(entry.get("note", "") or entry.get("comment", "") or "")

    # Try to extract abstract from note field
    abs_m = re.search(r"[Aa]bstract[^\d]*(\d{4}-\d{2}-\d{2})", note)
    abstracts = [abs_m.group(1)] if abs_m else []

    return {
        "name":      f"{name} '{year[-2:]}" if year else name,
        "full":      str(entry.get("description", "") or "").strip(),
        "tier":      tier,
        "year":      year,
        "deadlines": _norm_deadlines(entry.get("deadline")),
        "abstracts": abstracts,
        "venue":     str(entry.get("place", "") or "").strip(),
        "date":      str(entry.get("date", "") or "").strip(),
        "url":       str(entry.get("link", "") or "").strip(),
        "note":      note,
        "_source":   "remote",
    }

def _from_ai(entry: dict) -> dict | None:
    name = str(entry.get("title", "")).strip()
    tier = _match_tier(name)
    if not tier:
        return None
    year = str(entry.get("year", "")).strip()
    deadline = _extract_date(str(entry.get("deadline", "") or ""))
    abstract = _extract_date(str(entry.get("abstract_deadline", "") or ""))
    return {
        "name":      f"{name} '{year[-2:]}" if year else name,
        "full":      str(entry.get("full_name", "") or "").strip(),
        "tier":      tier,
        "year":      year,
        "deadlines": [deadline] if deadline else [],
        "abstracts": [abstract] if abstract else [],
        "venue":     str(entry.get("place", "") or "").strip(),
        "date":      str(entry.get("date", "") or "").strip(),
        "url":       str(entry.get("link", "") or "").strip(),
        "note":      str(entry.get("note", "") or "").strip(),
        "_source":   "remote",
    }

# ── Fetch ─────────────────────────────────────────────────────────────────────
def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "conf-deadline-tracker/3.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")

# ── Merge logic ───────────────────────────────────────────────────────────────
def _upsert(merged: dict, conf: dict):
    """
    Upsert a conference entry.
    Manual entries always win on all fields.
    Remote entries only fill in missing deadlines/fields.
    """
    key = _dedup_key(conf["name"], conf["year"])
    if key not in merged:
        merged[key] = dict(conf)
        return

    ex = merged[key]
    is_manual_existing = ex.get("_source") == "manual"
    is_manual_incoming = conf.get("_source") == "manual"

    if is_manual_incoming and not is_manual_existing:
        # Manual overwrites remote
        merged[key] = dict(conf)
        return

    if is_manual_existing and not is_manual_incoming:
        # Remote only fills gaps
        for dl in conf["deadlines"]:
            if dl not in ex["deadlines"]:
                ex["deadlines"].append(dl)
        if conf.get("abstracts") and not ex.get("abstracts"):
            ex["abstracts"] = conf["abstracts"]
        return

    # Same priority — merge deadlines
    for dl in conf["deadlines"]:
        if dl not in ex["deadlines"]:
            ex["deadlines"].append(dl)
    for ab in conf.get("abstracts", []):
        if ab not in ex.get("abstracts", []):
            ex.setdefault("abstracts", []).append(ab)

# ── Main crawl ────────────────────────────────────────────────────────────────
def crawl() -> list[dict]:
    merged: dict[str, dict] = {}

    # 1. Manual file (highest priority)
    if MANUAL_FILE.exists():
        print(f"  Reading {MANUAL_FILE.name}…")
        entries = _parse_yaml(MANUAL_FILE.read_text(encoding="utf-8"))
        ok = 0
        for e in entries:
            c = _from_manual(e)
            if c:
                _upsert(merged, c)
                ok += 1
        print(f"  → {ok} manual entries loaded")
    else:
        print(f"  WARN: {MANUAL_FILE} not found — skipping manual entries")

    # 2. Remote YAML sources
    for src in REMOTE_SOURCES:
        print(f"  Fetching {src['name']}…")
        try:
            text = _fetch(src["url"])
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        entries = _parse_yaml(text)
        print(f"  Parsed {len(entries)} entries")
        fn = _from_ai if src["fmt"] == "ai" else _from_sec
        matched = 0
        for e in entries:
            c = fn(e)
            if c:
                _upsert(merged, c)
                matched += 1
        print(f"  → {matched} matched")

    # 3. Sort deadlines within each entry
    for c in merged.values():
        c["deadlines"] = sorted(set(c["deadlines"]))
        c["abstracts"] = sorted(set(c.get("abstracts", [])))
        c.pop("_source", None)

    # 4. Global sort
    return sorted(
        merged.values(),
        key=lambda x: (
            TIER_ORDER.get(x["tier"], 99),
            x["deadlines"][0] if x["deadlines"] else "9999-99-99",
            x["name"],
        ),
    )

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("Conference Deadline Crawler")
    print("=" * 50)
    confs = crawl()
    print(f"\nTotal: {len(confs)} conferences")
    for c in confs:
        print(f"  [{c['tier']:4}] {c['name']:<24} deadlines={c['deadlines']}")

    payload = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "conferences": confs,
    }
    OUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten → {OUT_FILE}")

if __name__ == "__main__":
    main()
