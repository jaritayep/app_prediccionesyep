"""
Football-Data.co.uk — World Cup 2026 Scraper
=============================================
Downloads the official WC2026 XLSX from football-data.co.uk,
filters to June 17–23 2026, and exports a single clean CSV.
"""

import io
import sys
import requests
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

XLSX_URL   = "https://www.football-data.co.uk/WorldCup2026.xlsx"
DATE_FROM  = datetime(2026, 6, 17)
DATE_TO    = datetime(2026, 6, 23)
OUTPUT_CSV = "worldcup_jun17_23.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.football-data.co.uk/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
}

# Preferred column order for output
PREFERRED_COLS = [
    "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "Referee", "Attendance",
    "HS", "AS", "HST", "AST",
    "HC", "AC", "HF", "AF",
    "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
    "PSH",   "PSD",   "PSA",
    "WHH",   "WHD",   "WHA",
    "MaxH",  "MaxD",  "MaxA",
    "AvgH",  "AvgD",  "AvgA",
    "B365>2.5", "B365<2.5",
    "P>2.5",    "P<2.5",
    "Max>2.5",  "Max<2.5",
    "Avg>2.5",  "Avg<2.5",
    "AHh", "B365AHH", "B365AHA",
    "MaxAHH", "MaxAHA", "AvgAHH", "AvgAHA",
]


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_xlsx(url: str) -> bytes:
    print(f"  → Descargando: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"  ✗ HTTP {resp.status_code}")
        sys.exit(1)
    print(f"  ✓ Descargado ({len(resp.content):,} bytes)")
    return resp.content


# ─────────────────────────────────────────────────────────────────────────────
# PARSE & FILTER
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter(raw: bytes) -> pd.DataFrame:
    xl = pd.ExcelFile(io.BytesIO(raw))
    print(f"  → Hojas encontradas: {xl.sheet_names}")

    # ── Print actual columns of each sheet so we can see what we're working with
    print("\n  📋 Diagnóstico de columnas:")
    for sheet in xl.sheet_names:
        try:
            peek = xl.parse(sheet, nrows=2, dtype=str)
            print(f"     '{sheet}': {list(peek.columns)}")
        except Exception as e:
            print(f"     '{sheet}': error — {e}")
    print()

    # ── Use only the main 2026 sheet (not qualifiers or past tournaments) ─────
    target_sheet = None
    for sheet in xl.sheet_names:
        name_lower = sheet.lower()
        if "2026" in name_lower and "qualifier" not in name_lower:
            target_sheet = sheet
            break

    if target_sheet is None:
        # Fallback: just use the first sheet
        target_sheet = xl.sheet_names[0]
        print(f"  ⚠ No se encontró hoja 2026 sin qualifiers — usando '{target_sheet}'")

    print(f"  → Usando hoja: '{target_sheet}'")
    df = xl.parse(target_sheet, dtype=str)

    # ── Detect date / team columns flexibly ───────────────────────────────────
    cols_lower = {c.strip().lower(): c for c in df.columns}

    def find_col(candidates):
        for candidate in candidates:
            if candidate.lower() in cols_lower:
                return cols_lower[candidate.lower()]
        # Fallback: partial match
        for candidate in candidates:
            for key, real in cols_lower.items():
                if candidate.lower() in key:
                    return real
        return None

    date_col = find_col(["Date", "date", "Fecha"])
    home_col = find_col(["HomeTeam", "Home Team", "Home", "hometeam"])
    away_col = find_col(["AwayTeam", "Away Team", "Away", "awayteam"])

    if date_col is None:
        print(f"  ✗ No se encontró columna de fecha. Columnas disponibles: {list(df.columns)}")
        sys.exit(1)
    if home_col is None:
        print(f"  ✗ No se encontró columna HomeTeam. Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    print(f"  → Columnas detectadas: Date='{date_col}', Home='{home_col}', Away='{away_col}'")

    # Normalise column names
    rename = {}
    if date_col != "Date":     rename[date_col] = "Date"
    if home_col != "HomeTeam": rename[home_col] = "HomeTeam"
    if away_col and away_col != "AwayTeam": rename[away_col] = "AwayTeam"
    if rename:
        df = df.rename(columns=rename)

    print(f"  → Total filas: {len(df)}")

    # ── Parse dates ───────────────────────────────────────────────────────────
    def parse_date(s):
        s = str(s).strip()
        for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass
        # Try pandas as last resort
        try:
            return pd.to_datetime(s, dayfirst=True)
        except Exception:
            return pd.NaT

    df["_date_parsed"] = df["Date"].apply(parse_date)

    # Show sample of parsed dates for debugging
    sample = df[["Date", "_date_parsed"]].dropna().head(5)
    print(f"\n  → Muestra de fechas parseadas:\n{sample.to_string(index=False)}\n")

    df = df.dropna(subset=["_date_parsed"])

    # ── Filter June 17–23 ─────────────────────────────────────────────────────
    mask = (df["_date_parsed"] >= DATE_FROM) & (df["_date_parsed"] <= DATE_TO)
    filtered = df[mask].copy()
    filtered = filtered.drop(columns=["_date_parsed"])

    # Sort by date + time
    if "Time" in filtered.columns:
        filtered["_sort_dt"] = pd.to_datetime(
            filtered["Date"].astype(str) + " " + filtered["Time"].fillna("00:00").astype(str),
            dayfirst=True, errors="coerce"
        )
        filtered = filtered.sort_values("_sort_dt").drop(columns=["_sort_dt"])
    else:
        filtered = filtered.sort_values("Date")

    print(f"  → Partidos del 17–23 junio: {len(filtered)}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ORDERING
# ─────────────────────────────────────────────────────────────────────────────

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    leading  = [c for c in PREFERRED_COLS if c in df.columns]
    trailing = [c for c in df.columns if c not in PREFERRED_COLS]
    return df[leading + trailing]


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  ⚽ WC2026 SCRAPER — football-data.co.uk")
    print(f"  📅 Rango: {DATE_FROM.strftime('%d/%m/%Y')} → {DATE_TO.strftime('%d/%m/%Y')}")
    print("=" * 65)

    raw = download_xlsx(XLSX_URL)
    df  = load_and_filter(raw)

    if df.empty:
        print("\n⚠ No se encontraron partidos en el rango 17–23 junio.")
        print("  Puede que el XLSX aún no incluya esas fechas o el formato de fecha es distinto.")
        sys.exit(0)

    df = reorder_columns(df)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*65}")
    print(f"✅ CSV guardado: {OUTPUT_CSV}")
    print(f"   Partidos: {len(df)}  |  Columnas: {len(df.columns)}")
    print("=" * 65)

    # Quick preview
    preview_cols = ["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG",
                    "B365H", "B365D", "B365A"]
    available = [c for c in preview_cols if c in df.columns]
    if available:
        print("\n📋 Vista previa:")
        print(df[available].to_string(index=False))


if __name__ == "__main__":
    main()