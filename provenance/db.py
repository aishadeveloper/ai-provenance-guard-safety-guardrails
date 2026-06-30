"""Centralized SQLite access.

A single ``Database`` object owns the connection settings and is the **one place** a
connection is created — instead of every data-access function opening its own ad-hoc
connection. It is built once (in the app factory) and passed to the data-access
modules (audit, certificates, analytics).

``transaction()`` groups statements into one atomic unit (commit on success, roll
back on any error, always close), which is what lets multi-step operations — e.g.
reading a row and inserting another based on it — happen atomically rather than as
two separate connections that could interleave under concurrent requests.

For SQLite this per-operation-connection model is appropriate; the point of the
abstraction is that connection creation, row factory, and pragmas live in one place,
so swapping in a connection pool or a request-scoped session later is a localized
change rather than an edit to every function.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator


class Database:
    """Owns SQLite connection settings and hands out managed connections."""

    def __init__(self, path: str):
        self.path = path

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection; commit on success, roll back on error, always close.

        Every statement executed inside one ``with db.transaction() as conn:`` block
        is applied atomically.
        """
        conn = self._open()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a read connection, always closed at the end (no implicit commit)."""
        conn = self._open()
        try:
            yield conn
        finally:
            conn.close()
