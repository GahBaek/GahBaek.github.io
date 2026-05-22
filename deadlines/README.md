# Conference Deadline Tracker

학회 마감일 자동 수집 + GitHub Pages 트래커

## 구조

```
├── index.html                 ← GitHub Pages 프론트엔드
├── data.json                  ← 크롤러가 자동 생성 (커밋됨)
├── conferences.manual.yml     ← ✏️  직접 편집하는 파일
├── crawler/
│   └── crawl.py               ← Python 크롤러 (외부 패키지 없음)
└── .github/workflows/
    └── update.yml             ← 매일 자동 실행
```

## 데이터 소스 (우선순위 순)

| 우선순위 | 소스 | 설명 |
|---------|------|------|
| 1 (최우선) | `conferences.manual.yml` | 직접 관리 |
| 2 | [sec-deadlines](https://github.com/sec-deadlines/sec-deadlines.github.io) | 보안 |
| 3 | [se-deadlines](https://github.com/se-deadlines/se-deadlines.github.io) | 소프트웨어 공학 |
| 4 | [ai-deadlines](https://github.com/abhshkdz/ai-deadlines) | AI/ML |

manual이 최우선이므로, 자동 소스 데이터가 틀렸어도 수동으로 덮어쓸 수 있습니다.

## 트래킹 대상

| 카테고리 | 학회 |
|---------|------|
| **Sys1** | OSDI, SOSP, EuroSys, ATC, NSDI |
| **SE1** | ICSE, FSE |
| **SE2** | ASE |
| **AI1** | IJCAI, AAAI, WWW |
| **Sec1** | S&P, CCS, USENIX Security, NDSS |
| **Sec2** | ACSAC, RAID, ASIACCS, ESORICS, Euro S&P, DSN |
| **Sec3** | CODASPY, DIMVA, SecureComm |
| **Sec4** | SAC, IFIP-SEC |

## GitHub Pages 설치

```bash
# 1. 이 파일들을 레포에 push
git add .
git commit -m "init"
git push

# 2. Settings → Pages → Source: main / (root)
# 3. Settings → Actions → General → Workflow permissions: Read and write
# 4. Actions 탭 → "Update Conference Deadlines" → Run workflow  (첫 실행)
```

이후 매일 KST 09:00 자동 실행됩니다.

---

## 컨퍼런스 수동 추가/수정

`conferences.manual.yml`을 편집하세요. 크롤러가 다음 실행 시 반영됩니다.

### 형식

```yaml
- name: NewConf          # 약어 (연도 제외)
  tier: Sec2             # Sys1 / SE1 / SE2 / AI1 / Sec1 / Sec2 / Sec3 / Sec4
  year: 2027
  description: Full Conference Name
  link: https://newconf2027.example.com/cfp
  deadline:
    - "2026-10-01 23:59"   # AoE 기준
  abstract_deadline:
    - "2026-09-24 23:59"
  date: Mar 2027
  place: City, Country
  note: 비고 (옵션)
```

### 멀티 사이클 (Cycle 1, 2)

```yaml
- name: S&P
  tier: Sec1
  year: 2027
  deadline:
    - "2026-06-11 23:59"   # Cycle 1
    - "2026-11-17 23:59"   # Cycle 2
  abstract_deadline:
    - "2026-06-04 23:59"   # Cycle 1 abstract
    - "2026-11-10 23:59"   # Cycle 2 abstract
  ...
```

### 즉시 반영하기

```bash
python3 crawler/crawl.py   # data.json 재생성
git add data.json
git commit -m "update deadlines"
git push
```

또는 Actions 탭 → "Update Conference Deadlines" → **Run workflow**

---

## 로컬 미리보기

```bash
python3 crawler/crawl.py   # data.json 생성
python3 -m http.server     # http://localhost:8000
```

> `file://`로 직접 열면 fetch가 막히므로 반드시 http.server를 사용하세요.
