#!/usr/bin/env python3
"""
Conference Deadline Crawler v5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sources (priority):
  1. conferences.manual.yml
  2. casys-kaist  (ARCH,SYS,ML,OTHER,TBD — all sub categories)
  3. sec-deadlines
  4. se-deadlines
  5. mlciv  (all conference files that match whitelist)
  6. abhshkdz/ai-deadlines (fallback)

Filter: conference year >= today.year - 2
"""

import json, re, urllib.request
from datetime import datetime, timezone, date
from pathlib import Path

ROOT        = Path(__file__).parent.parent
MANUAL_FILE = ROOT / "conferences.manual.yml"
OUT_FILE    = ROOT / "data.json"
TODAY       = date.today()
YEAR_MIN = TODAY.year - 2
YEAR_MAX = TODAY.year + 2 

SRC_RANK = {
    "manual": 0,
    "casys": 1,
    "sec": 2,
    "se": 3,
    "mlciv": 4,
    "abhshkdz": 5,
    "remote": 9,
}

TIER_ORDER = {t:i for i,t in enumerate(
    ["Sys","SE","AI","Sec"])}

# ── Whitelist ─────────────────────────────────────────────────────────────────
WHITELIST = [
    # Sys1
    (r"^osdi\b",            "Sys"),
    (r"^sosp\b",            "Sys"),
    (r"^eurosys\b",         "Sys"),
    (r"^atc\b",             "Sys"),
    (r"^nsdi\b",            "Sys"),
    # SE1
    (r"^icse\b(?!.*(?:seip|seis|nier|t&tb|jf\b|ict|ds\b|src|nfs|ip\b|ivr))", "SE"),
    (r"^fse\b(?!.*(?:ip\b|ivr|jf\b))",  "SE"),
    (r"^esec/fse\b",        "SE"),
    # SE2
    (r"^ase\b(?!.*(?:nier|td\b|jf\b|is\b|src|ds\b))", "SE"),
    # AI1
    (r"^ijcai\b",           "AI"),
    (r"^ijcai-ecai\b",      "AI"),
    (r"^aaai\b(?!\s+\d{4}\s+(?:spring|fall|winter|summer))", "AI"),
    (r"^www\b",             "AI"),
    # Sec1
    (r"^s&p\b",             "Sec"),
    (r"^ieee s&p\b",        "Sec"),
    (r"^ccs\b(?!.*(?:workshop|ccsw|\sw\b))", "Sec"),
    (r"^usenix sec",        "Sec"),
    (r"^ndss\b",            "Sec"),
    # Sec2
    (r"^acsac\b",           "Sec"),
    (r"^raid\b",            "Sec"),
    (r"^asiaccs\b",         "Sec"),
    (r"^asia ccs\b",        "Sec"),
    (r"^esorics\b",         "Sec"),
    (r"^euro s&p\b",        "Sec"),
    (r"^eurosp\b",          "Sec"),
    (r"^dsn\b",             "Sec"),
    # Sec3
    (r"^codaspy\b",         "Sec"),
    (r"^dimva\b",           "Sec"),
    (r"^securecomm\b",      "Sec"),
    # Sec4
    (r"^sac\b(?!.*(?:mat|sec\b|net|ped|svt|dac|dlt|dsp))", "Sec"),
    (r"^ifip.?sec\b",       "Sec"),
]
_COMPILED = [(re.compile(p, re.I), t) for p, t in WHITELIST]

def match_tier(name: str) -> str | None:
    n = name.lower().strip()
    for pat, tier in _COMPILED:
        if pat.match(n):
            return tier
    return None

# ── Dedup key ─────────────────────────────────────────────────────────────────
_ALIASES = {
    "ifip sec":"IFIP-SEC","ifip-sec":"IFIP-SEC",
    "asia ccs":"ASIACCS","asiaccs":"ASIACCS",
    "s&p (oakland)":"S&P","ieee s&p":"S&P",
    "usenix security":"USENIX Security",
    "esec/fse":"FSE",
    "euro s&p":"Euro S&P","eurosp":"Euro S&P",
    "ijcai-ecai":"IJCAI",
}
def _canon(name: str) -> str:
    n = name.lower().strip()
    for alias, canon in _ALIASES.items():
        if n.startswith(alias): return canon.lower()
    return re.split(r"[\s'(\-]", n)[0]

def _key(name: str, year: str) -> str:
    return f"{_canon(name)}|{year}"

# ── Date helpers ──────────────────────────────────────────────────────────────
def _xdate(s) -> str | None:
    if not s: return None
    cur = datetime.now(timezone.utc).year
    s = str(s).replace("%Y", str(cur)).replace("%y", str(cur+1))
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None

def _xlist(raw) -> list[str]:
    if not raw: return []
    items = [raw] if isinstance(raw, str) else list(raw)
    return [d for i in items if (d := _xdate(str(i)))]

def _has_future_dl(c: dict) -> bool:
    """True if TBD or has at least one future deadline."""
    dls = c.get("deadlines", [])
    if not dls:
        return True   # TBD — keep
    for dl in dls:
        try:
            y, m, d = map(int, dl.split("-"))
            if date(y, m, d) >= TODAY:
                return True
        except:
            pass
    return False

# ── YAML parser ───────────────────────────────────────────────────────────────
def _parse(text: str) -> list[dict]:
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
            if line.strip().startswith("#"): i+=1; continue
            m = re.match(r'^[-\s]\s*([a-zA-Z_]+):\s*(.*)', line)
            if not m: i+=1; continue
            key, val = m.group(1).strip(), m.group(2).strip()
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                entry[key] = [x.strip().strip('"').strip("'")
                              for x in re.split(r",\s*",inner) if x.strip()] if inner.strip() else []
                i+=1; continue
            sub, j = [], i+1
            while j < len(lines) and re.match(r'^\s{2,}-\s', lines[j]):
                item = re.sub(r'^\s+-\s*','',lines[j]).strip().strip('"').strip("'")
                if not item.startswith("#"): sub.append(item)
                j+=1
            if sub: entry[key]=sub; i=j; continue
            entry[key] = val.strip('"').strip("'"); i+=1
        if entry: entries.append(entry)
    return entries

def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent":"conf-deadline-tracker/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")

# ── Build conf dict ───────────────────────────────────────────────────────────
def _make(name,full,tier,year,deadlines,abstracts,venue,date_s,url,note,src):
    yr = str(year).strip()
    display = f"{name} '{yr[-2:]}" if yr and len(yr)>=2 else name
    return {
        "name":      display,
        "full":      (full or "").strip(),
        "tier":      tier,
        "year":      yr,
        "deadlines": sorted(set(deadlines)),
        "abstracts": sorted(set(abstracts)),
        "venue":     (venue or "").strip(),
        "date":      (date_s or "").strip(),
        "url":       (url or "").strip(),
        "note":      re.sub(r'<[^>]+>','', note or '').strip(),
        "_src":      src,
    }

def _from_manual(e):
    name = str(e.get("name","")).strip()
    tier = str(e.get("tier","")).strip()
    if not name or not tier: return None
    yr = str(e.get("year","")).strip()
    return _make(name, e.get("description",""), tier, yr,
                 _xlist(e.get("deadline",[])), _xlist(e.get("abstract_deadline",[])),
                 e.get("place",""), e.get("date",""), e.get("link",""),
                 e.get("note",""), "manual")

def _from_ai(e, src="remote"):
    name = str(e.get("title","")).strip()
    tier = match_tier(name)
    if not tier: return None
    yr  = str(e.get("year","")).strip()
    dl  = _xdate(str(e.get("deadline","") or ""))
    ab  = _xdate(str(e.get("abstract_deadline","") or ""))
    return _make(name, e.get("full_name", e.get("description","")),
                 tier, yr,
                 [dl] if dl else [], [ab] if ab else [],
                 e.get("place",""), e.get("date",""), e.get("link",""),
                 e.get("note",""), src)

def _from_sec(e):
    name = str(e.get("name","")).strip()
    tier = match_tier(name)
    if not tier: return None
    yr   = str(e.get("year","")).strip()
    note = str(e.get("note","") or e.get("comment","") or "")
    ab_m = re.search(r"[Aa]bstract[^\d]*(\d{4}-\d{2}-\d{2})", note)
    return _make(name, e.get("description",""), tier, yr,
                 _xlist(e.get("deadline",[])), [ab_m.group(1)] if ab_m else [],
                 e.get("place",""), e.get("date",""), e.get("link",""),
                 note, "remote")

# ── Upsert ────────────────────────────────────────────────────────────────────
def _upsert(merged, conf):
    k = _key(conf["name"], conf["year"])

    if k not in merged:
        merged[k] = dict(conf)
        return

    ex = merged[k]
    ex_rank = SRC_RANK.get(ex.get("_src"), 99)
    in_rank = SRC_RANK.get(conf.get("_src"), 99)

    # higher-priority source wins completely
    if in_rank < ex_rank:
        merged[k] = dict(conf)
        return

    # lower-priority source must not modify higher-priority data
    if in_rank > ex_rank:
        return

    # same-priority source only: merge deadlines
    ex["deadlines"] = sorted(set(ex.get("deadlines", []) + conf.get("deadlines", [])))
    ex["abstracts"] = sorted(set(ex.get("abstracts", []) + conf.get("abstracts", [])))

def _in_year_window(c: dict) -> bool:
    try:
        y = int(str(c.get("year", "")).strip())
    except ValueError:
        return False

    return YEAR_MIN <= y <= YEAR_MAX

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_manual(merged):
    if not MANUAL_FILE.exists():
        print(f"  WARN: {MANUAL_FILE.name} not found"); return
    ok = 0
    for e in _parse(MANUAL_FILE.read_text(encoding="utf-8")):
        c = _from_manual(e)
        if c: _upsert(merged, c); ok += 1
    print(f"  → {ok} manual entries")

def load_casys(merged):
    # All sub categories: ARCH, SYS, ML, OTHER, TBD
    base = "https://raw.githubusercontent.com/casys-kaist/casys-kaist.github.io/gh-pages/_data/conferences"
    total = 0
    for yr in range(YEAR_MIN, TODAY.year + 3):
        try: text = _fetch(f"{base}/{yr}.yml")
        except: continue
        matched = sum(1 for e in _parse(text)
                      if (c := _from_ai(e, "casys")) and (_upsert(merged, c) or True))
        total += matched
        print(f"    casys {yr}.yml → {matched} matched")
    print(f"  → {total} total from casys-kaist")

def load_sec(merged):
    url = "https://raw.githubusercontent.com/sec-deadlines/sec-deadlines.github.io/master/_data/conferences.yml"
    text = _fetch(url)
    entries = _parse(text)
    ok = sum(1 for e in entries if (c:=_from_sec(e)) and (_upsert(merged,c) or True))
    print(f"  → {ok} matched ({len(entries)} parsed)")

def load_se(merged):
    url = "https://raw.githubusercontent.com/se-deadlines/se-deadlines.github.io/master/_data/conferences.yml"
    text = _fetch(url)
    entries = _parse(text)
    ok = sum(1 for e in entries if (c:=_from_sec(e)) and (_upsert(merged,c) or True))
    print(f"  → {ok} matched ({len(entries)} parsed)")

def load_mlciv(merged):
    # Fetch ALL yml files from mlciv and filter by whitelist
    base = "https://raw.githubusercontent.com/mlciv/ai-deadlines/gh-pages/_data/conferences"
    # Full file list from repo archive (cached list for efficiency)
    files = [
        "aaai","ijcai","www","iclr","icml","neurips","cvpr","iccv","eccv",
        "acl","emnlp","naacl","coling","aaai","kdd","sigir","wsdm","mm",
        "icra","iros","rss","aistats","uai","aamas","chi","cscw","sigmod",
        "icde","sigdial","interspeech","icassp","bmvc","wacv","siggraph",
        "siggraphasia","ecml","acml","colm","recsys","umap","edm","lak",
        "3dv","cvc","icmla","icmlcn","icaaaaiml","aied","log","starsem",
    ]
    total = 0
    for fname in files:
        try: text = _fetch(f"{base}/{fname}.yml")
        except: continue
        matched = sum(1 for e in _parse(text)
                      if (c:=_from_ai(e,"mlciv")) and (_upsert(merged,c) or True))
        if matched: total += matched
    print(f"  → {total} total from mlciv")

def load_abhshkdz(merged):
    url = "https://raw.githubusercontent.com/abhshkdz/ai-deadlines/gh-pages/_data/conferences.yml"
    text = _fetch(url)
    entries = _parse(text)
    ok = sum(1 for e in entries if (c:=_from_ai(e,"abhshkdz")) and (_upsert(merged,c) or True))
    print(f"  → {ok} matched ({len(entries)} parsed)")

# ── Main ──────────────────────────────────────────────────────────────────────
def crawl():
    merged = {}
    print("1. Manual");       load_manual(merged)
    print("2. casys-kaist");
    try: load_casys(merged)
    except Exception as e: print(f"  ERROR: {e}")
    print("3. sec-deadlines");
    try: load_sec(merged)
    except Exception as e: print(f"  ERROR: {e}")
    print("4. se-deadlines");
    try: load_se(merged)
    except Exception as e: print(f"  ERROR: {e}")
    print("5. mlciv");
    try: load_mlciv(merged)
    except Exception as e: print(f"  ERROR: {e}")
    print("6. abhshkdz (fallback)");
    try: load_abhshkdz(merged)
    except Exception as e: print(f"  ERROR: {e}")

    result = []
    for c in merged.values():
        c["deadlines"] = sorted(set(c["deadlines"]))
        c["abstracts"] = sorted(set(c.get("abstracts",[])))
        c.pop("_src", None)
        if _has_future_dl(c):
            result.append(c)

    return sorted(result, key=lambda x:(
        TIER_ORDER.get(x["tier"],99),
        x["deadlines"][0] if x["deadlines"] else "9999",
        x["name"],
    ))

def main():
    print(f"{'='*50}\n Conference Deadline Crawler v5\n Filter: future deadlines only\n{'='*50}")
    confs = crawl()
    print(f"\nTotal: {len(confs)}")
    for c in confs:
        print(f"  [{c['tier']:4}] {c['name']:<28} {c['deadlines']}")
    OUT_FILE.write_text(
        json.dumps({"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "conferences": confs}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nWritten → {OUT_FILE}")

if __name__ == "__main__":
    main()
