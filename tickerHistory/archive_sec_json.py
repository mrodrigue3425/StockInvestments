import json
import hashlib
import sqlite3
from datetime import datetime, timezone

import requests


URL = "https://www.sec.gov/files/company_tickers_exchange.json"
DB_PATH = "sec_archive.db"

HEADERS = {
    "User-Agent": "MySecArchive your-email@example.com"
}


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT NOT NULL,
            url TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            raw_json TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            cik INTEGER,
            name TEXT,
            ticker TEXT,
            exchange TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_companies_snapshot_id
        ON companies(snapshot_id)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_companies_ticker
        ON companies(ticker)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_companies_cik
        ON companies(cik)
    """)

    conn.commit()
    conn.close()


def fetch_sec_data():
    response = requests.get(URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def save_snapshot(raw_text):
    conn = get_connection()
    cur = conn.cursor()

    fetched_at = datetime.now(timezone.utc).isoformat()
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    # check if we already have this exact version
    cur.execute("SELECT id FROM snapshots WHERE content_hash = ?", (content_hash,))
    existing = cur.fetchone()

    if existing:
        conn.close()
        print("No change. This snapshot already exists.")
        return None

    cur.execute("""
        INSERT INTO snapshots (fetched_at, url, content_hash, raw_json)
        VALUES (?, ?, ?, ?)
    """, (fetched_at, URL, content_hash, raw_text))

    snapshot_id = cur.lastrowid

    payload = json.loads(raw_text)
    fields = payload["fields"]

    rows_to_insert = []
    for row in payload["data"]:
        record = dict(zip(fields, row))
        rows_to_insert.append((
            snapshot_id,
            record.get("cik"),
            record.get("name"),
            record.get("ticker"),
            record.get("exchange"),
        ))

    cur.executemany("""
        INSERT INTO companies (snapshot_id, cik, name, ticker, exchange)
        VALUES (?, ?, ?, ?, ?)
    """, rows_to_insert)

    conn.commit()
    conn.close()

    print(f"Saved new snapshot {snapshot_id} with {len(rows_to_insert)} companies.")
    return snapshot_id


def main():
    init_db()
    raw_text = fetch_sec_data()
    save_snapshot(raw_text)


if __name__ == "__main__":
    main()