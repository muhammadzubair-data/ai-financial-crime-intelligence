"""Case management: a small SQLite store for investigation decisions, notes and an audit trail.

SQLite keeps the demo self-contained. The interface (open, update, add note, history) is what a
real case-management system would expose, so swapping the backend later only touches this file.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

STATUSES = ["Open", "In review", "Escalated", "Closed"]
DISPOSITIONS = ["Undecided", "Suspicious - recommend SAR filing", "False positive", "Keep monitoring",
                "Insufficient information"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL UNIQUE,
    alert_id TEXT, priority TEXT, risk REAL,
    status TEXT NOT NULL DEFAULT 'Open',
    disposition TEXT NOT NULL DEFAULT 'Undecided',
    assigned_to TEXT NOT NULL DEFAULT '',
    opened_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes (
    note_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER NOT NULL, created_at TEXT NOT NULL,
    author TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER NOT NULL, at TEXT NOT NULL,
    actor TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CaseStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(":memory:") if self.path == ":memory:" else None
        with closing(self._conn()) as con:
            con.executescript(SCHEMA)
            con.commit()

    def _conn(self) -> sqlite3.Connection:
        if self._mem is not None:
            return _NoClose(self._mem)
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _audit(self, con, case_id: int, actor: str, action: str, detail: str) -> None:
        con.execute("INSERT INTO audit (case_id, at, actor, action, detail) VALUES (?,?,?,?,?)",
                    (case_id, _now(), actor or "unknown", action, detail))

    # ------------------------------------------------------------------ write
    def open_case(self, account_id: int, alert_id: str = "", priority: str = "", risk: float = 0.0,
                  actor: str = "analyst") -> int:
        """Open a case for an account, or return the existing one (one case per account)."""
        existing = self.by_account(account_id)
        if existing:
            return existing["case_id"]
        with closing(self._conn()) as con:
            now = _now()
            cur = con.execute("INSERT INTO cases (account_id, alert_id, priority, risk, opened_at, updated_at) "
                              "VALUES (?,?,?,?,?,?)", (int(account_id), alert_id, priority, float(risk), now, now))
            self._audit(con, cur.lastrowid, actor, "opened", f"Case opened from alert {alert_id}".strip())
            con.commit()
            return cur.lastrowid

    def update(self, case_id: int, actor: str = "analyst", **changes) -> None:
        allowed = {"status": STATUSES, "disposition": DISPOSITIONS, "assigned_to": None}
        current = self.get(case_id)
        if current is None:
            raise KeyError(f"case {case_id} not found")
        with closing(self._conn()) as con:
            for field, value in changes.items():
                if field not in allowed:
                    raise ValueError(f"cannot update field {field!r}")
                if allowed[field] is not None and value not in allowed[field]:
                    raise ValueError(f"{value!r} is not a valid {field}")
                if current[field] != value:
                    con.execute(f"UPDATE cases SET {field} = ?, updated_at = ? WHERE case_id = ?", (value, _now(), case_id))
                    self._audit(con, case_id, actor, f"{field} changed", f"{current[field] or '(empty)'} -> {value or '(empty)'}")
            con.commit()

    def add_note(self, case_id: int, author: str, body: str) -> None:
        body = (body or "").strip()
        if not body:
            raise ValueError("note is empty")
        with closing(self._conn()) as con:
            con.execute("INSERT INTO notes (case_id, created_at, author, body) VALUES (?,?,?,?)",
                        (case_id, _now(), author or "analyst", body))
            con.execute("UPDATE cases SET updated_at = ? WHERE case_id = ?", (_now(), case_id))
            self._audit(con, case_id, author, "note added", body[:80])
            con.commit()

    # ------------------------------------------------------------------ read
    def _one(self, sql: str, args: tuple) -> dict | None:
        with closing(self._conn()) as con:
            cur = con.execute(sql, args)
            row = cur.fetchone()
            return dict(zip([c[0] for c in cur.description], row)) if row else None

    def _frame(self, sql: str, args: tuple = ()) -> pd.DataFrame:
        with closing(self._conn()) as con:
            cur = con.execute(sql, args)
            return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

    def get(self, case_id: int) -> dict | None:
        return self._one("SELECT * FROM cases WHERE case_id = ?", (int(case_id),))

    def by_account(self, account_id: int) -> dict | None:
        return self._one("SELECT * FROM cases WHERE account_id = ?", (int(account_id),))

    def notes(self, case_id: int) -> pd.DataFrame:
        return self._frame("SELECT created_at, author, body FROM notes WHERE case_id = ? ORDER BY note_id", (int(case_id),))

    def history(self, case_id: int) -> pd.DataFrame:
        return self._frame("SELECT at, actor, action, detail FROM audit WHERE case_id = ? ORDER BY audit_id", (int(case_id),))

    def list(self) -> pd.DataFrame:
        return self._frame("SELECT * FROM cases ORDER BY updated_at DESC")


class _NoClose:
    """Lets an in-memory connection be used with ``closing`` without actually closing it."""

    def __init__(self, con):
        self._con = con

    def __getattr__(self, name):
        return getattr(self._con, name)

    def close(self):
        pass
