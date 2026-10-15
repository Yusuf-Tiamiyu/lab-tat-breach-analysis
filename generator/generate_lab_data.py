"""Synthetic diagnostic-lab turnaround-time (TAT) data.

Produces six months of test-level records for a fictional nine-site lab network,
in two shapes:

  data/tests.csv            one tidy row per test (used by the analysis notebook
                            and the Power BI kit)
  data/reference_times.csv  the test catalogue with target TATs
  data/sites.csv            site attributes
  exports/*.xlsx            weekly files in the same column layout as a LIS
                            export (used by the TAT platform demo)

Nothing here is real. Site names, test volumes and timings are invented. The
process rules are modelled on how a typical multi-site diagnostic lab works, so
that the patterns an analyst would look for actually exist in the data:

  * samples entered outside opening hours wait for the site to open
  * a 2pm-4pm shift handover slows work (improved after week 9, see below)
  * referred samples travel by scheduled courier before testing starts
  * immunoassay and molecular tests run in fixed daily batches
  * two sites are short-staffed, so busy days queue up
  * instruments occasionally go down for hours at a time
  * weekend staffing is thinner

Week 9 is the "dashboard launch": from then on, handover delays and verification
lag shrink gradually, which is what moves compliance from the mid-50s to ~70%.

Run:  python generator/generate_lab_data.py            (writes data/ and exports/)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20250106
START = date(2025, 1, 6)          # a Monday
WEEKS = 26
LAUNCH_WEEK = 9                   # 1-based week the improvement starts

ROOT = Path(__file__).resolve().parent.parent
DATA, EXPORTS = ROOT / "data", ROOT / "exports"

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Site:
    code: int            # performing-site code, as a LIS would print it
    prefix: str          # requisition-number prefix for samples collected here
    name: str
    is_hub: bool         # performs the full test menu
    is_24h: bool
    daily_reqs: float    # average requisitions per weekday
    capacity: float      # tests per open hour the bench can absorb before a queue forms
    courier_hours: float # travel time to the nearest hub
    staffing: float      # 1.0 = fully staffed; lower means slower handling


SITES = [
    Site(8101, "8101", "Central",    True,  True,  210, 16, 0.0, 1.0),
    Site(8102, "8102", "Harbour",    True,  True,  150, 12, 1.0, 1.0),
    Site(8103, "8103", "Northgate",  False, True,   90,  7, 1.5, 1.0),
    Site(8104, "8104", "Riverside",  False, False,  85,  6, 2.0, 0.6),   # short-staffed
    Site(8105, "8105", "Eastfield",  False, False,  70,  7, 2.5, 1.0),
    Site(8106, "8106", "Hilltop",    False, False,  55,  6, 3.0, 1.0),
    Site(8107, "8107", "Lakeside",   False, False,  65,  5, 2.0, 0.65),  # short-staffed
    Site(8108, "8108", "Westbrook",  False, False,  45,  5, 4.0, 1.0),
    Site(8109, "8109", "Parkview",   False, False,  40,  5, 3.5, 1.0),
]
HUB_FOR = {s.name: ("Central" if s.courier_hours in (0.0, 1.5, 2.0, 2.5) else "Harbour") for s in SITES}
HUB_FOR["Harbour"] = "Harbour"
BY_NAME = {s.name: s for s in SITES}

# ---------------------------------------------------------------------------
# Test catalogue: (name, department, target hours, popularity, hub_only)
# ---------------------------------------------------------------------------
CATALOGUE = [
    ("Full Blood Count",            "Haematology",    4,  30, False),
    ("ESR",                         "Haematology",    4,   6, False),
    ("Blood Film",                  "Haematology",   24,   4, False),
    ("Coagulation Profile",         "Haematology",    6,   4, True),
    ("Haemoglobin Genotype",        "Haematology",   24,   5, False),
    ("Blood Group",                 "Haematology",    4,   6, False),
    ("Fasting Blood Sugar",         "Chemistry",      4,  14, False),
    ("Electrolytes, Urea, Creatinine", "Chemistry",   6,  18, False),
    ("Liver Function Test",         "Chemistry",      6,  14, False),
    ("Lipid Profile",               "Chemistry",      6,  12, False),
    ("HbA1c",                       "Chemistry",     24,   9, False),
    ("Uric Acid",                   "Chemistry",      6,   5, False),
    ("Serum Calcium",               "Chemistry",      6,   4, False),
    ("Thyroid Function Test",       "Immunoassay",   24,  10, True),
    ("PSA",                         "Immunoassay",   24,   6, True),
    ("Vitamin D",                   "Immunoassay",   48,   5, True),
    ("Prolactin",                   "Immunoassay",   24,   3, True),
    ("Testosterone",                "Immunoassay",   24,   3, True),
    ("Troponin I",                  "Immunoassay",    2,   2, True),
    ("HIV Screening",               "Serology",       6,   9, False),
    ("Hepatitis B Surface Antigen", "Serology",       6,   9, False),
    ("Hepatitis C Antibody",        "Serology",      24,   6, False),
    ("Widal Test",                  "Serology",       6,   7, False),
    ("Malaria Parasite",            "Serology",       4,  12, False),
    ("H. pylori Antigen",           "Serology",      24,   3, False),
    ("Urinalysis",                  "Microbiology",   4,  10, False),
    ("Urine Culture",               "Microbiology",  72,   7, True),
    ("Blood Culture",               "Microbiology", 120,   3, True),
    ("Stool Culture",               "Microbiology",  72,   3, True),
    ("High Vaginal Swab Culture",   "Microbiology",  72,   3, True),
    ("Hepatitis B Viral Load",      "Molecular",     96,   2, True),
    ("HIV Viral Load",              "Molecular",     96,   2, True),
    ("SARS-CoV-2 PCR",              "Molecular",     24,   2, True),
]
CAT = pd.DataFrame(CATALOGUE, columns=["test_name", "department", "target_hours", "weight", "hub_only"])
CAT["p"] = CAT["weight"] / CAT["weight"].sum()

# Departments that run in fixed batches: list of start hours per day
BATCH_RUNS = {"Immunoassay": [10, 16], "Molecular": [13]}
# Typical analytical time once a test is on the instrument (hours)
RUN_HOURS = {"Haematology": 0.4, "Chemistry": 0.8, "Immunoassay": 1.5, "Serology": 0.7,
             "Microbiology": 0.5, "Molecular": 6.0}
COURIER_DEPARTURES = [11, 15, 19]


def week_no(d: date) -> int:
    return (d - START).days // 7 + 1


def improvement(d: date) -> float:
    """0 before launch, rising to 1 over the eight weeks after it."""
    w = week_no(d)
    return float(np.clip((w - LAUNCH_WEEK + 1) / 8, 0, 1))


def is_open(site: Site, ts: datetime) -> bool:
    if site.is_24h:
        return True
    dow, h = ts.weekday(), ts.hour + ts.minute / 60
    if dow == 6:
        return False
    if dow == 5:
        return 8 <= h < 14
    return 8 <= h < 20


def next_open(site: Site, ts: datetime) -> datetime:
    t = ts
    for _ in range(24 * 4 * 4):
        if is_open(site, t):
            return t
        t = (t + timedelta(minutes=15)).replace(second=0, microsecond=0)
    return t


def next_batch(dept: str, ts: datetime, site: Site) -> datetime:
    hours = BATCH_RUNS[dept]
    day = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    for add in range(0, 5):
        for h in hours:
            cand = day + timedelta(days=add, hours=h)
            if cand >= ts and is_open(site, cand) and not (cand.weekday() == 6 and not site.is_24h):
                return cand
    return ts


def entry_time(site: Site, d: date) -> datetime:
    """Collection/entry time: morning peak, smaller afternoon peak, some night work at 24h sites."""
    r = rng.random()
    if site.is_24h and r < 0.12:
        h = rng.choice([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]) + rng.random()
    elif r < 0.70:
        h = rng.normal(10.0, 1.6)
    else:
        h = rng.normal(15.5, 1.8)
    h = float(np.clip(h, 0, 23.98))
    if d.weekday() == 5 and not site.is_24h:
        h = float(np.clip(rng.normal(10, 1.2), 7.5, 13.9))
    return datetime(d.year, d.month, d.day) + timedelta(hours=h)


def shift_of(hour: int, site: Site) -> str:
    if 8 <= hour < 14:
        return "Morning (8am-2pm)"
    if 14 <= hour < 16:
        return "Handover (2pm-4pm)"
    if 16 <= hour < 20:
        return "Afternoon (4pm-8pm)"
    return "Night (8pm-8am)" if site.is_24h else "Outside Hours"


def downtime_windows() -> dict:
    """Random instrument outages: {(site, dept): [(start, end), ...]}."""
    out = {}
    days = WEEKS * 7
    for s in SITES:
        for dept in RUN_HOURS:
            n = rng.poisson(days / 60)
            wins = []
            for _ in range(n):
                start = datetime.combine(START, datetime.min.time()) + timedelta(
                    days=int(rng.integers(0, days)), hours=float(rng.uniform(7, 18)))
                wins.append((start, start + timedelta(hours=float(rng.uniform(3, 20)))))
            out[(s.name, dept)] = wins
    return out


DOWNTIME = downtime_windows()


def after_downtime(site: str, dept: str, ts: datetime) -> datetime:
    for a, b in DOWNTIME.get((site, dept), []):
        if a <= ts < b:
            return b
    return ts


def simulate() -> pd.DataFrame:
    rows = []
    req_seq = {s.name: 100000 for s in SITES}
    days = [START + timedelta(days=i) for i in range(WEEKS * 7)]
    dow_factor = [1.05, 1.0, 1.0, 1.0, 0.95, 0.55, 0.25]

    for d in days:
        imp = improvement(d)
        growth = 1 + 0.004 * week_no(d)
        for site in SITES:
            if d.weekday() == 6 and not site.is_24h:
                continue
            n_reqs = rng.poisson(site.daily_reqs * dow_factor[d.weekday()] * growth / 2.6)
            # Daily load ratio (tests arriving / tests the bench can absorb) drives queueing
            load = n_reqs * 2.3 / (site.capacity * (12 if not site.is_24h else 16))
            for _ in range(n_reqs):
                req_seq[site.name] += 1
                req_no = f"{site.prefix}{req_seq[site.name]:06d}"
                ent = entry_time(site, d)
                k = int(min(6, 1 + rng.poisson(1.3)))
                tests = CAT.sample(n=k, replace=False, weights="p", random_state=int(rng.integers(1e9)))
                for t in tests.itertuples():
                    perf = BY_NAME[HUB_FOR[site.name]] if (t.hub_only and not site.is_hub) else site
                    shipped = perf.name != site.name
                    rec = None
                    if shipped:
                        dep_hour = next((h for h in COURIER_DEPARTURES if h >= ent.hour + 0.5), None)
                        depart = (ent.replace(hour=dep_hour, minute=0) if dep_hour is not None
                                  else (ent + timedelta(days=1)).replace(hour=COURIER_DEPARTURES[0], minute=0))
                        if not is_open(site, depart):
                            depart = next_open(site, depart)
                        rec = depart + timedelta(hours=site.courier_hours + abs(rng.normal(0, 0.4)))
                    start = rec if shipped else ent
                    start = next_open(perf, start)
                    urgent = t.target_hours <= 2

                    # Pre-analytical handling (sorting, centrifuging, accessioning); shrinks after launch
                    pre_h = rng.gamma(2.0, 1.12) * (1 - 0.38 * imp) / perf.staffing
                    # Queue at the bench: grows with load, worse at short-staffed sites and weekends
                    queue_h = rng.gamma(1.5, 0.5) * (1 + 2.5 * max(0.0, load - 0.7)) / perf.staffing
                    if d.weekday() >= 5:
                        queue_h *= 1.5
                    if urgent:
                        pre_h, queue_h = pre_h * 0.15, 0.0
                    start += timedelta(hours=pre_h + queue_h)

                    if t.department in BATCH_RUNS:
                        start = next_batch(t.department, start, perf)
                    start = after_downtime(perf.name, t.department, start)

                    run_h = RUN_HOURS[t.department] * rng.uniform(0.8, 1.4)
                    if t.department == "Microbiology" and t.test_name != "Urinalysis":
                        run_h = t.target_hours * rng.uniform(0.62, 0.95)
                    done = start + timedelta(hours=run_h)

                    # Handover: work finishing between 2pm and 4pm waits for sign-off
                    if 14 <= done.hour < 16:
                        done += timedelta(minutes=float(rng.gamma(2.0, 60 * (1 - 0.7 * imp))))
                    # Verification lag; shrinks after the launch
                    done += timedelta(minutes=float(rng.exponential(85 * (1 - 0.5 * imp)) * (0.3 if urgent else 1)))
                    done = next_open(perf, done) if not perf.is_24h else done

                    rows.append({
                        "req_number": req_no, "origin_site": site.name, "perf_site": perf.name,
                        "perf_site_code": perf.code, "test_name": t.test_name, "department": t.department,
                        "entered_at": ent, "received_at": rec, "verified_at": done,
                        "target_hours": float(t.target_hours),
                    })
    df = pd.DataFrame(rows)

    # ---- realistic data-quality noise -------------------------------------
    n = len(df)
    pending = rng.random(n) < 0.008                      # never verified in the export window
    df.loc[pending, "verified_at"] = pd.NaT
    zero = rng.random(n) < 0.004                         # verified at the same minute as entry
    df.loc[zero, "verified_at"] = df.loc[zero, "entered_at"]
    no_dept = rng.random(n) < 0.003
    df.loc[no_dept, "department"] = None
    uncatalogued = rng.random(n) < 0.002                 # test missing from the reference sheet
    df.loc[uncatalogued, "target_hours"] = np.nan
    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["entered_at", "req_number", "test_name"]).reset_index(drop=True)
    df.insert(0, "test_id", np.arange(1, len(df) + 1))
    df["tat_hours"] = (df["verified_at"] - df["entered_at"]).dt.total_seconds() / 3600
    df["tat_from_receipt_hours"] = (df["verified_at"] - df["received_at"]).dt.total_seconds() / 3600
    df["is_shipped"] = df["received_at"].notna()
    df["entry_date"] = df["entered_at"].dt.date
    df["entry_hour"] = df["entered_at"].dt.hour
    df["weekday"] = df["entered_at"].dt.day_name()
    df["week"] = ((df["entered_at"].dt.normalize() - pd.Timestamp(START)).dt.days // 7 + 1).astype(int)
    df["shift"] = [shift_of(h, BY_NAME[s]) for h, s in zip(df["entry_hour"], df["origin_site"])]
    measurable = df["tat_hours"].notna() & df["target_hours"].notna()
    df["breached"] = np.where(measurable, df["tat_hours"] > df["target_hours"], np.nan)
    return df


def write_exports(df: pd.DataFrame) -> None:
    """Weekly workbooks in the LIS export layout: dates as Excel serials, durations as fractions of a day."""
    EXPORTS.mkdir(exist_ok=True)
    epoch = pd.Timestamp("1899-12-30")

    def serial(s):
        return (s - epoch).dt.total_seconds() / 86400

    ref = CAT[["test_name", "target_hours", "department"]].rename(
        columns={"test_name": "Test Name", "department": "Department"})
    ref["Target TAT"] = ref.pop("target_hours") / 24
    ref = ref[["Test Name", "Target TAT", "Department"]]

    for w, g in df.groupby("week"):
        first = START + timedelta(days=7 * (w - 1))
        last = first + timedelta(days=6)
        out = pd.DataFrame({
            "PerfSite": g["perf_site_code"],
            "Req Number": g["req_number"].astype(np.int64),
            "Test Name": g["test_name"],
            "Department": g["department"],
            "EntDate & EntTime": serial(g["entered_at"]),
            "SiteRecDate & SiteRecTime": serial(g["received_at"]),
            "TestVerDate & TestVerTime": serial(g["verified_at"]),
            "Ver-Ent": g["tat_hours"] / 24,
            "Ver-Rec": g["tat_from_receipt_hours"] / 24,
            "Target TAT": g["target_hours"] / 24,
        })
        name = f"TAT Data ({first:%d %b} - {last:%d %b %Y}).xlsx"
        with pd.ExcelWriter(EXPORTS / name) as xw:
            out.to_excel(xw, sheet_name="TAT Data", index=False)
            ref.to_excel(xw, sheet_name="Reference_Times", index=False)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    df = derive(simulate())
    cols = ["test_id", "req_number", "origin_site", "perf_site", "test_name", "department",
            "entered_at", "received_at", "verified_at", "target_hours", "tat_hours",
            "tat_from_receipt_hours", "is_shipped", "entry_date", "entry_hour", "weekday",
            "week", "shift", "breached"]
    df[cols].to_csv(DATA / "tests.csv", index=False, float_format="%.4f", date_format="%Y-%m-%d %H:%M")
    CAT[["test_name", "department", "target_hours", "hub_only"]].to_csv(DATA / "reference_times.csv", index=False)
    pd.DataFrame([{"site": s.name, "code": s.code, "is_hub": s.is_hub, "is_24h": s.is_24h,
                   "refers_to": HUB_FOR[s.name]} for s in SITES]).to_csv(DATA / "sites.csv", index=False)
    write_exports(df)

    m = df["breached"].notna()
    by_week = df[m].groupby("week")["breached"].mean().mul(100).round(1)
    print(f"{len(df):,} tests, {df['req_number'].nunique():,} requisitions, {WEEKS} weeks")
    print("compliance by week (%):", (100 - by_week).round(1).tolist())


if __name__ == "__main__":
    main()
