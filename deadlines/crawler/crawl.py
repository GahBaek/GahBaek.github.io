#!/usr/bin/env python3
"""
Conference Deadline Crawler v4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sources (priority order):
  1. conferences.manual.yml  — user-maintained (highest priority)
  2. casys-kaist/casys-kaist.github.io  — Systems (OSDI/SOSP/ATC/NSDI/EuroSys)
  3. sec-deadlines/sec-deadlines.github.io — Security
  4. se-deadlines/se-deadlines.github.io  — SE
  5. mlciv/ai-deadlines  — AI/ML (per-conference YAML files)
  6. abhshkdz/ai-deadlines — AI/ML fallback

Rules:
  - Only show conferences from 2 years ago to future
  - Manual entries override remote on all fields
  - Remote entries only fill in missing deadlines
"""

import json, re, urllib.request
from datetime import datetime, timezone, date
from pathlib import Path

ROOT        = Path(__file__).parent.parent
MANUAL_FILE = ROOT / "conferences.manual.yml"
OUT_FILE    = ROOT / "data.json"

TODAY      = date.today()
YEAR_MIN   = TODAY.year - 2   # hide anything older than 2 years

TIER_ORDER = {t: i for i, t in enumerate(
    ["Sys1","SE1","SE2","AI1","Sec1","Sec2","Sec3","Sec4"]
)}

# ── Target whitelist ──────────────────────────────────────────────────────────
# (title_pattern, tier)   — matched against lowercase title/name
WHITELIST = [
    # Sys1
    (r"^osdi\b",            "Sys1"),
    (r"^sosp\b",            "Sys1"),
    (r"^eurosys\b",         "Sys1"),
    (r"^atc\b",             "Sys1"),
    (r"^nsdi\b",            "Sys1"),
    # SE1
    (r"^icse\b(?!.*(?:seip|seis|nier|t&tb|jf\b|ict|ds\b|src|nfs|ip\b|ivr))", "SE1"),
    (r"^fse\b(?!.*(?:ip\b|ivr|jf\b))",  "SE1"),
    (r"^esec/fse\b",        "SE1"),
    # SE2
    (r"^ase\b(?!.*(?:nier|td\b|jf\b|is\b|src|ds\b))", "SE2"),
    # AI1
    (r"^ijcai\b",           "AI1"),
    (r"^aaai\b(?!\s+\d{4}\s+(?:spring|fall|winter|summer))", "AI1"),
    (r"^www\b",             "AI1"),
    # Sec1
    (r"^s&p\b",             "Sec1"),
    (r"^ieee s&p\b",        "Sec1"),
    (r"^ccs\b(?!.*(?:workshop|ccsw|\sw\b))", "Sec1"),
    (r"^usenix sec",        "Sec1"),
    (r"^ndss\b",            "Sec1"),
    # Sec2
    (r"^acsac\b",           "Sec2"),
    (r"^raid\b",            "Sec2"),
    (r"^asiaccs\b",         "Sec2"),
    (r"^asia ccs\b",        "Sec2"),
    (r"^esorics\b",         "Sec2"),
    (r"^euro s&p\b",        "Sec2"),
    (r"^eurosp\b",          "Sec2"),
    (r"^dsn\b",             "Sec2"),
    # Sec3
    (r"^codaspy\b",         "Sec3"),
    (r"^dimva\b",           "Sec3"),
    (r"^securecomm\b",      "Sec3"),
    # Sec4
    (r"^sac\b(?!.*(?:mat|sec\b|net|ped|svt|dac|dlt|dsp))", "Sec4"),
    (r"^ifip.?sec\b",       "Sec4"),
]
_COMPILED = [(re.compile(p, re.I), t) for p, t in WHITELIST]

def match_tier(name: str) -> str | None:
    n = name.lower().strip()
    for pat, tier in _COMPILED:
        if pat.match(n):
            return tier
    return None

# ── Name normalisation for dedup ──────────────────────────────────────────────
_ALIASES = {
    "ifip sec":         "IFIP-SEC",
    "ifip-sec":         "IFIP-SEC",
    "asia ccs":         "ASIACCS",
    "asiaccs":          "ASIACCS",
    "s&p (oakland)":    "S&P",
    "ieee s&p":         "S&P",
    "usenix security":  "USENIX Security",
    "esec/fse":         "FSE",
    "euro s&p":         "Euro S&P",
    "eurosp":           "Euro S&P",
    "ijcai-ecai":       "IJCAI",
}

def _canon(name: str) -> str:
    n = name.lower().strip()
    for alias, canon in _ALIASES.items():
        if n.startswith(alias):
            return canon.lower()
    return re.split(r"[\s'(\-]", n)[0]

def _dedup_key(name: str, year: str) -> str:
    return f"{_canon(name)}|{year}"

# ── Date helpers ──────────────────────────────────────────────────────────────
def _extract_date(s) -> str | None:
    if not s: return None
    cur = datetime.now(timezone.utc).year
    s = str(s).replace("%Y", str(cur)).replace("%y", str(cur+1))
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None

def _norm_list(raw) -> list[str]:
    if not raw: return []
    items = [raw] if isinstance(raw, str) else list(raw)
    return [d for item in items if (d := _extract_date(str(item)))]

def _conf_year(c: dict) -> int:
    """Determine the conference year for age filtering."""
    yr = str(c.get("year","")).strip()
    try:
        return int(yr)
    except ValueError:
        pass
    # fallback: latest deadline year
    for dl in c.get("deadlines",[]):
        m = re.match(r"(\d{4})", dl)
        if m: return int(m.group(1))
    return TODAY.year

def _is_too_old(c: dict) -> bool:
    return _conf_year(c) < YEAR_MIN

# ── YAML parser (stdlib) ──────────────────────────────────────────────────────
def _parse_yaml(text: str) -> list[dict]:
    """
    Parse sec-deadlines / casys-kaist / mlciv YAML list format.
    Handles: scalars, block lists, inline lists.
    """
    blocks = re.split(r'\n(?=- (?:name|title):\s)', "\n" + text)
    entries = []
    for block in blocks:
        block = block.strip()
        if not block or block.startswith("#"): continue
        entry: dict = {}
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("#"):
                i += 1; continue
            m = re.match(r'^[-\s]\s*([a-zA-Z_]+):\s*(.*)', line)
            if not m: i += 1; continue
            key = m.group(1).strip()
            val = m.group(2).strip()
            # inline list
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                entry[key] = [x.strip().strip('"').strip("'")
                              for x in re.split(r",\s*", inner) if x.strip()] if inner.strip() else []
                i += 1; continue
            # block list
            sub, j = [], i+1
            while j < len(lines) and re.match(r'^\s{2,}-\s', lines[j]):
                item = re.sub(r'^\s+-\s*','', lines[j]).strip().strip('"').strip("'")
                if not item.startswith("#"):
                    sub.append(item)
                j += 1
            if sub:
                entry[key] = sub; i = j; continue
            # scalar
            entry[key] = val.strip('"').strip("'")
            i += 1
        if entry: entries.append(entry)
    return entries

# ── Fetch ─────────────────────────────────────────────────────────────────────
def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "conf-deadline-tracker/4.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")

# ── Entry converters ──────────────────────────────────────────────────────────
def _make_conf(name, full, tier, year, deadlines, abstracts, venue, date_str, url, note, source):
    yr = str(year).strip()
    display = f"{name} '{yr[-2:]}" if yr and len(yr) >= 2 else name
    return {
        "name":      display,
        "full":      (full or "").strip(),
        "tier":      tier,
        "year":      yr,
        "deadlines": sorted(set(deadlines)),
        "abstracts": sorted(set(abstracts)),
        "venue":     (venue or "").strip(),
        "date":      (date_str or "").strip(),
        "url":       (url or "").strip(),
        "note":      re.sub(r'<[^>]+>', '', note or '').strip(),  # strip HTML tags
        "_source":   source,
    }

def _from_manual(e: dict) -> dict | None:
    name = str(e.get("name","")).strip()
    tier = str(e.get("tier","")).strip()
    if not name or not tier: return None
    year = str(e.get("year","")).strip()
    return _make_conf(
        name, e.get("description",""), tier, year,
        _norm_list(e.get("deadline",[])),
        _norm_list(e.get("abstract_deadline",[])),
        e.get("place",""), e.get("date",""), e.get("link",""),
        e.get("note",""), "manual"
    )

def _from_ai_fmt(e: dict, source: str = "remote") -> dict | None:
    """Parse ai-deadlines / casys-kaist / mlciv format (title field)."""
    name = str(e.get("title","")).strip()
    tier = match_tier(name)
    if not tier: return None
    year = str(e.get("year","")).strip()
    dl  = _extract_date(str(e.get("deadline","") or ""))
    ab  = _extract_date(str(e.get("abstract_deadline","") or ""))
    return _make_conf(
        name,
        e.get("full_name", e.get("description","")),
        tier, year,
        [dl] if dl else [],
        [ab] if ab else [],
        e.get("place",""), e.get("date",""), e.get("link",""),
        e.get("note",""), source
    )

def _from_sec_fmt(e: dict) -> dict | None:
    """Parse sec-deadlines / se-deadlines format (name field)."""
    name = str(e.get("name","")).strip()
    tier = match_tier(name)
    if not tier: return None
    year = str(e.get("year","")).strip()
    note = str(e.get("note","") or e.get("comment","") or "")
    # Try to extract abstract from note
    ab_match = re.search(r"[Aa]bstract[^\d]*(\d{4}-\d{2}-\d{2})", note)
    abstracts = [ab_match.group(1)] if ab_match else []
    return _make_conf(
        name, e.get("description",""), tier, year,
        _norm_list(e.get("deadline",[])),
        abstracts,
        e.get("place",""), e.get("date",""), e.get("link",""),
        note, "remote"
    )

# ── Merge ─────────────────────────────────────────────────────────────────────
def _upsert(merged: dict, conf: dict):
    key = _dedup_key(conf["name"], conf["year"])
    if key not in merged:
        merged[key] = dict(conf); return

    ex = merged[key]
    is_manual_ex  = ex.get("_source")   == "manual"
    is_manual_inc = conf.get("_source") == "manual"

    if is_manual_inc and not is_manual_ex:
        merged[key] = dict(conf); return   # manual overwrites remote

    if is_manual_ex and not is_manual_inc:
        # remote only fills gaps
        for d in conf["deadlines"]:
            if d not in ex["deadlines"]: ex["deadlines"].append(d)
        if conf["abstracts"] and not ex["abstracts"]:
            ex["abstracts"] = conf["abstracts"]
        return

    # same priority — merge
    for d in conf["deadlines"]:
        if d not in ex["deadlines"]: ex["deadlines"].append(d)
    for a in conf["abstracts"]:
        if a not in ex.get("abstracts",[]): ex.setdefault("abstracts",[]).append(a)

# ── Sources ───────────────────────────────────────────────────────────────────
def load_manual(merged):
    if not MANUAL_FILE.exists():
        print(f"  WARN: {MANUAL_FILE.name} not found"); return
    entries = _parse_yaml(MANUAL_FILE.read_text(encoding="utf-8"))
    ok = 0
    for e in entries:
        c = _from_manual(e)
        if c: _upsert(merged, c); ok += 1
    print(f"  → {ok} manual entries")

def load_casys_kaist(merged):
    """casys-kaist per-year YAML files (2024–2027)."""
    base = "https://raw.githubusercontent.com/casys-kaist/casys-kaist.github.io/gh-pages/_data/conferences"
    total = 0
    for year in range(YEAR_MIN, TODAY.year + 3):
        url = f"{base}/{year}.yml"
        try:
            text = _fetch(url)
        except Exception as e:
            print(f"    skip {year}: {e}"); continue
        entries = _parse_yaml(text)
        matched = 0
        for e in entries:
            c = _from_ai_fmt(e, "casys")
            if c: _upsert(merged, c); matched += 1
        total += matched
        print(f"    {year}.yml → {matched} matched ({len(entries)} parsed)")
    print(f"  → {total} total from casys-kaist")

def load_sec_deadlines(merged):
    url = "https://raw.githubusercontent.com/sec-deadlines/sec-deadlines.github.io/master/_data/conferences.yml"
    text = _fetch(url)
    entries = _parse_yaml(text)
    ok = sum(1 for e in entries if (c := _from_sec_fmt(e)) and (_upsert(merged, c) or True))
    print(f"  → {ok} matched ({len(entries)} parsed)")

def load_se_deadlines(merged):
    url = "https://raw.githubusercontent.com/se-deadlines/se-deadlines.github.io/master/_data/conferences.yml"
    text = _fetch(url)
    entries = _parse_yaml(text)
    ok = sum(1 for e in entries if (c := _from_sec_fmt(e)) and (_upsert(merged, c) or True))
    print(f"  → {ok} matched ({len(entries)} parsed)")

def load_mlciv(merged):
    """mlciv per-conference YAML files for our AI1 targets."""
    base = "https://raw.githubusercontent.com/mlciv/ai-deadlines/gh-pages/_data/conferences"
    targets = ["aaai", "ijcai", "www"]
    total = 0
    for t in targets:
        url = f"{base}/{t}.yml"
        try:
            text = _fetch(url)
        except Exception as e:
            print(f"    skip {t}: {e}"); continue
        entries = _parse_yaml(text)
        matched = 0
        for e in entries:
            c = _from_ai_fmt(e, "mlciv")
            if c: _upsert(merged, c); matched += 1
        total += matched
        print(f"    {t}.yml → {matched} matched")
    print(f"  → {total} total from mlciv")

def load_abhshkdz(merged):
    """Fallback: abhshkdz/ai-deadlines single YAML."""
    url = "https://raw.githubusercontent.com/abhshkdz/ai-deadlines/gh-pages/_data/conferences.yml"
    text = _fetch(url)
    entries = _parse_yaml(text)
    ok = sum(1 for e in entries if (c := _from_ai_fmt(e, "abhshkdz")) and (_upsert(merged, c) or True))
    print(f"  → {ok} matched ({len(entries)} parsed)")

# ── Main ──────────────────────────────────────────────────────────────────────
def crawl() -> list[dict]:
    merged: dict[str, dict] = {}

    print("1. Manual entries")
    load_manual(merged)

    print("2. casys-kaist (Systems)")
    try: load_casys_kaist(merged)
    except Exception as e: print(f"  ERROR: {e}")

    print("3. sec-deadlines (Security)")
    try: load_sec_deadlines(merged)
    except Exception as e: print(f"  ERROR: {e}")

    print("4. se-deadlines (SE)")
    try: load_se_deadlines(merged)
    except Exception as e: print(f"  ERROR: {e}")

    print("5. mlciv (AI)")
    try: load_mlciv(merged)
    except Exception as e: print(f"  ERROR: {e}")

    print("6. abhshkdz/ai-deadlines (AI fallback)")
    try: load_abhshkdz(merged)
    except Exception as e: print(f"  ERROR: {e}")

    # Finalise
    result = []
    for c in merged.values():
        c["deadlines"] = sorted(set(c["deadlines"]))
        c["abstracts"] = sorted(set(c.get("abstracts",[])))
        c.pop("_source", None)
        if not _is_too_old(c):
            result.append(c)

    return sorted(result, key=lambda x: (
        TIER_ORDER.get(x["tier"], 99),
        x["deadlines"][0] if x["deadlines"] else "9999-99-99",
        x["name"],
    ))

def main():
    print("=" * 52)
    print(" Conference Deadline Crawler v4")
    print(f" Filter: {YEAR_MIN} – present")
    print("=" * 52)
    confs = crawl()
    print(f"\nTotal: {len(confs)} conferences")
    for c in confs:
        print(f"  [{c['tier']:4}] {c['name']:<26} {c['deadlines']}")

    OUT_FILE.write_text(
        json.dumps({"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "conferences": confs}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\nWritten → {OUT_FILE}")

if __name__ == "__main__":
    main()
