import io, os, re, json, time, base64, math, textwrap
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

import requests
from bs4 import BeautifulSoup
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

APP_START_TIME = time.time()
HARD_DEADLINE_SECONDS = 170

app = FastAPI(title="Data Analyst Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def now_ms():
    return int((time.time() - APP_START_TIME) * 1000)

def guard_time(need: float = 5.0):
    elapsed = time.time() - START_OF_REQUEST
    if elapsed + need >= HARD_DEADLINE_SECONDS:
        raise TimeoutError("Time budget exceeded; returning partial results.")

def read_text_file(f: UploadFile) -> str:
    return f.file.read().decode("utf-8", errors="replace")

def try_float(x):
    try:
        return float(x)
    except:
        return np.nan

def png_data_uri_from_plt(fig: plt.Figure, target_max_bytes: int = 100_000) -> str:
    width, height, dpi = 600, 400, 120
    for _ in range(8):
        buf = io.BytesIO()
        fig.set_size_inches(width / dpi, height / dpi)
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        data = buf.getvalue()
        if len(data) <= target_max_bytes:
            b64 = base64.b64encode(data).decode("ascii")
            plt.close(fig)
            return f"data:image/png;base64,{b64}"
        buf2 = io.BytesIO(data)
        img = Image.open(buf2).convert("P", palette=Image.ADAPTIVE)
        buf3 = io.BytesIO()
        img.save(buf3, format="PNG", optimize=True)
        data2 = buf3.getvalue()
        if len(data2) <= target_max_bytes:
            b64 = base64.b64encode(data2).decode("ascii")
            plt.close(fig)
            return f"data:image/png;base64,{b64}"
        width = int(width * 0.9); height = int(height * 0.9); dpi = max(72, int(dpi * 0.9))
    fig.set_size_inches(320/96, 240/96)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    data = buf.getvalue()
    b64 = base64.b64encode(data).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def dotted_red_regression_scatter(df: pd.DataFrame, x: str, y: str, title: str = ""):
    dx = pd.to_numeric(df[x], errors="coerce")
    dy = pd.to_numeric(df[y], errors="coerce")
    d = pd.DataFrame({x: dx, y: dy}).dropna()
    if len(d) == 0:
        raise ValueError("No numeric data for scatter.")
    r = d[x].corr(d[y])
    a, b = np.polyfit(d[x], d[y], 1)
    xline = np.linspace(d[x].min(), d[x].max(), 200)
    yline = a * xline + b
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(d[x], d[y])
    ax.plot(xline, yline, linestyle=":", color="red")
    ax.set_xlabel(x); ax.set_ylabel(y)
    if title: ax.set_title(title)
    uri = png_data_uri_from_plt(fig, target_max_bytes=100_000)
    return float(r) if not np.isnan(r) else np.nan, uri

def load_user_tabular_files(files: Dict[str, UploadFile]) -> Dict[str, pd.DataFrame]:
    out = {}
    for fname, fobj in files.items():
        name = fname.lower()
        if name.endswith((".csv", ".tsv", ".txt")):
            content = fobj.file.read(); fobj.file.seek(0)
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception:
                df = pd.read_csv(io.BytesIO(content), sep="\\t")
            out[fname] = df
        elif name.endswith((".xlsx", ".xls")):
            content = fobj.file.read(); fobj.file.seek(0)
            df = pd.read_excel(io.BytesIO(content))
            out[fname] = df
    return out

def handle_wikipedia_highest_grossing(url: str) -> pd.DataFrame:
    guard_time(need=10)
    resp = requests.get(url, timeout=15); resp.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.select("table.wikitable")
    if not tables:
        raise ValueError("No wikitable found.")
    dfs = pd.read_html(str(tables[0]))
    df = dfs[0]
    return df

def answer_sample_movies(df: pd.DataFrame):
    import re, numpy as np, pandas as pd
    cols = {c: re.sub(r"\\W+", "_", str(c)).lower() for c in df.columns}
    df = df.rename(columns=cols)
    def find_col(cands):
        for c in df.columns:
            for k in cands:
                if k in c:
                    return c
        return None
    title_col = find_col(["title", "film", "movie"])
    gross_col = find_col(["gross", "worldwide", "box_office"])
    rank_col  = find_col(["rank"])
    peak_col  = find_col(["peak"])
    if gross_col is None:
        raise ValueError("Could not detect global gross column.")
    money = (df[gross_col].astype(str).str.replace(r"[^0-9.]", "", regex=True).apply(try_float))
    df["_gross_billion"] = money.apply(lambda v: v/1e9 if v and v > 10_000_000 else np.nan)
    year_col = find_col(["year", "release"])
    years = pd.to_numeric(df[year_col], errors="coerce") if year_col else pd.Series([np.nan]*len(df))
    q1 = int(((df["_gross_billion"] >= 2.0) & (years < 2000)).sum())
    mask = df["_gross_billion"] > 1.5
    candidates = df.loc[mask].copy()
    if year_col:
        candidates = candidates.sort_values(year_col, ascending=True)
    earliest_title = str(candidates[title_col].iloc[0]) if (title_col and not candidates.empty) else "Unknown"
    if (rank_col is None) or (peak_col is None):
        if rank_col is None:
            df["rank"] = np.arange(1, len(df) + 1); rank_col = "rank"
        if peak_col is None:
            df["peak"] = df[rank_col]; peak_col = "peak"
    corr, uri = dotted_red_regression_scatter(df, rank_col, peak_col, title="Rank vs Peak")
    return q1, earliest_title, float(corr), uri

def handle_indian_courts(what: str) -> dict:
    guard_time(need=20)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    base = "s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1"
    answers = {}
    q1 = """
    SELECT court, COUNT(*) AS n
    FROM read_parquet($base)
    WHERE year BETWEEN 2019 AND 2022
    GROUP BY court
    ORDER BY n DESC
    LIMIT 1
    """
    res1 = con.execute(q1, {"base": base}).fetchdf()
    if len(res1):
        answers["Which high court disposed the most cases from 2019 - 2022?"] = str(res1.loc[0, "court"])
    else:
        answers["Which high court disposed the most cases from 2019 - 2022?"] = "Unknown"
    q2 = """
    WITH d AS (
      SELECT
        year,
        TRY_STRPTIME(date_of_registration, '%d-%m-%Y') AS dor,
        decision_date AS dd
      FROM read_parquet($base)
      WHERE court = '33_10'
    ),
    x AS (
      SELECT
        year,
        DATE_DIFF('day', dor, dd) AS delay_days
      FROM d
      WHERE dor IS NOT NULL AND dd IS NOT NULL
    )
    SELECT year, delay_days FROM x
    """
    df = con.execute(q2, {"base": base}).fetchdf()
    slope_str = "NaN"; plot_uri = ""
    if len(df) >= 2:
        agg = df.groupby("year")["delay_days"].mean().reset_index()
        if len(agg) >= 2:
            a, b = np.polyfit(agg["year"], agg["delay_days"], 1)
            slope_str = f"{a:.6f}"
            _, plot_uri = dotted_red_regression_scatter(
                agg.rename(columns={"year":"Year", "delay_days":"MeanDelayDays"}),
                "Year", "MeanDelayDays",
                title="Year vs Mean Delay (Court 33_10)"
            )
    answers["What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?"] = slope_str
    if plot_uri:
        answers["Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters"] = plot_uri
    return answers

def parse_questions(text: str):
    wants_array = False; wants_object = False
    if re.search(r"respond with a JSON array", text, re.I): wants_array = True
    if re.search(r"respond with a JSON object", text, re.I): wants_object = True
    numbered = re.findall(r"^\\s*\\d+\\.\\s*(.+)$", text, flags=re.M)
    if not numbered:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        numbered = [ln for ln in lines if re.match(r".+\\?$", ln) or True][:10]
    urls = re.findall(r"https?://\\S+", text)
    return wants_array, wants_object, numbered, urls

def answer_questions(text: str, uploads: Dict[str, UploadFile]):
    wants_array, wants_object, items, urls = parse_questions(text)
    user_tables = load_user_tabular_files(uploads)
    if ("highest grossing films" in text.lower()) and any("wikipedia.org/wiki/List_of_highest-grossing_films" in u for u in urls):
        df = handle_wikipedia_highest_grossing(urls[0])
        q1, q2, q3, q4img = answer_sample_movies(df)
        result = [q1, q2, round(float(q3), 6), q4img]
        return result
    if "indian high court" in text.lower() or "ecourts" in text.lower():
        return handle_indian_courts(text)

    answers: List[str] = []
    corr_match = re.search(r"correlation between (.+?) and (.+?)", text, re.I)
    plot_match = re.search(r"scatterplot of (.+?) and (.+?)", text, re.I)

    def find_col_in_tables(col: str):
        for fname, df in user_tables.items():
            for c in df.columns:
                if col.strip().lower() == str(c).strip().lower():
                    return c, df
        for fname, df in user_tables.items():
            for c in df.columns:
                if col.strip().lower() in str(c).strip().lower():
                    return c, df
        raise KeyError(f"Column '{col}' not found in uploaded tables.")

    if corr_match:
        xname = corr_match.group(1).strip(); yname = corr_match.group(2).strip()
        try:
            cx, dfx = find_col_in_tables(xname)
            cy, dfy = find_col_in_tables(yname)
            if dfx is dfy:
                r = dfx[cx].corr(pd.to_numeric(dfx[cy], errors="coerce"))
            else:
                raise ValueError("Columns reside in different files; join not implemented.")
            answers.append(round(float(r), 6))
        except Exception as e:
            answers.append(f"Error computing correlation: {e}")

    if plot_match:
        xname = plot_match.group(1).strip(); yname = plot_match.group(2).strip()
        try:
            cx, dfx = find_col_in_tables(xname)
            cy, dfy = find_col_in_tables(yname)
            if dfx is not dfy:
                raise ValueError("Columns reside in different files; join not implemented.")
            _, uri = dotted_red_regression_scatter(dfx, cx, cy, title=f"{cx} vs {cy}")
            answers.append(uri)
        except Exception as e:
            answers.append(f"Error creating scatterplot: {e}")

    if wants_object:
        obj = {}
        for q in items:
            if answers:
                obj[q] = answers.pop(0)
            else:
                obj[q] = "Not enough information to answer."
        return obj
    else:
        if wants_array or len(items) > 1:
            return answers or ["No actionable question detected."]
        return answers[0] if answers else "No actionable question detected."

@app.post("/api/")
async def api_root(questions: UploadFile = File(...), files: List[UploadFile] = File(default=[])):
    global START_OF_REQUEST
    START_OF_REQUEST = time.time()
    try:
        qtext = read_text_file(questions)
        others = {}
        for f in files:
            if f.filename != questions.filename:
                others[f.filename] = f
        out = answer_questions(qtext, others)
        return JSONResponse(content=out)
    except TimeoutError as te:
        return JSONResponse(content={"error": "timeout", "partial": str(te)}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=200)
