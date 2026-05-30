"""SQLite-backed persistence for profiles, prompt history and globals.

The web UI used to keep prompt history (`[{text, fav}]`), favourite
prompts (the ``fav`` subset) and the globals template block in the
browser's ``localStorage``. This module moves all three server-side and
groups them under named **profiles** — each profile is an isolated set of
{prompt history, favourites, globals}.

One :class:`SqliteProfileStore` instance owns a single shared connection
(``check_same_thread=False``) guarded by a re-entrant lock, so the async
RPC handlers can call straight in without spawning threads — every method
is a sub-millisecond query. WAL mode keeps readers from blocking the
single writer.

Ordering of prompts is ``created_at DESC`` (newest first); the client
groups favourites above non-favourites for display, matching the previous
``localStorage`` behaviour. :meth:`SqliteProfileStore.push_prompt` upserts
(an existing text bumps to the front, keeping its ``fav`` flag) and caps
the non-favourite tail at :data:`NONFAV_CAP` per profile.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

from ..paths import DB_PATH, ensure_dirs, valid_profile_name

# Non-favourite prompts kept per profile. Favourites are never trimmed.
# Mirrors the old client-side ``LS.setRecents`` cap of 50.
NONFAV_CAP = 50

DEFAULT_PROFILE_NAME = "Default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT NOT NULL,
  globals    TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prompts (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  text       TEXT NOT NULL,
  fav        INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  UNIQUE(profile_id, text)
);
CREATE INDEX IF NOT EXISTS idx_prompts_profile ON prompts(profile_id);
-- Profile names double as on-disk output directory components, so they must
-- be unique (see zimt.paths.profile_out_dir).
CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_name ON profiles(name);
"""


class ProfileNotFound(Exception):
    """Raised when a profile id doesn't exist."""


def _clean_name(name: str) -> str:
    """Trim and validate a profile name; raise ValueError if unusable."""
    name = name.strip()
    if not name:
        raise ValueError("profile name must not be empty")
    if not valid_profile_name(name):
        raise ValueError(
            "profile name may contain only letters, digits, spaces, '.', '_', "
            "'-' and must not start with '.'"
        )
    return name


class SqliteProfileStore:
    """Profiles + prompt history + globals over a single SQLite file."""

    def __init__(self, db_path: str) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----- profiles -------------------------------------------------------

    def list_profiles(self) -> list[dict[str, Any]]:
        """All profiles with their prompt counts, oldest first.

        Auto-creates a single :data:`DEFAULT_PROFILE_NAME` profile when the
        table is empty so the UI always has at least one to select.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.id, p.name, "
                "  (SELECT COUNT(*) FROM prompts WHERE profile_id = p.id) AS prompt_count "
                "FROM profiles p ORDER BY p.id ASC"
            ).fetchall()
            if not rows:
                self._insert_profile(DEFAULT_PROFILE_NAME, "", time.time())
                rows = self._conn.execute(
                    "SELECT p.id, p.name, "
                    "  (SELECT COUNT(*) FROM prompts WHERE profile_id = p.id) AS prompt_count "
                    "FROM profiles p ORDER BY p.id ASC"
                ).fetchall()
            return [
                {"id": r["id"], "name": r["name"], "prompt_count": r["prompt_count"]}
                for r in rows
            ]

    def create_profile(self, name: str) -> dict[str, Any]:
        name = _clean_name(name)
        with self._lock:
            try:
                pid = self._insert_profile(name, "", time.time())
            except sqlite3.IntegrityError:
                raise ValueError(f"a profile named {name!r} already exists")
            self._conn.commit()
            return {"id": pid, "name": name, "prompt_count": 0}

    def ensure_profile(self, name: str) -> dict[str, Any]:
        """Create the profile if no profile with this name exists; idempotent.

        Used by the startup output-layout migration so a backing row always
        exists for the migrated ``OUT_DIR/Default/`` folder.
        """
        name = _clean_name(name)
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM profiles WHERE name = ?", (name,)
            ).fetchone()
            if row is not None:
                return {"id": row["id"], "name": name}
            pid = self._insert_profile(name, "", time.time())
            self._conn.commit()
            return {"id": pid, "name": name}

    def delete_profile(self, profile_id: int) -> None:
        with self._lock:
            self._require_profile(profile_id)
            (count,) = self._conn.execute("SELECT COUNT(*) FROM profiles").fetchone()
            if count <= 1:
                raise ValueError("cannot delete the last remaining profile")
            self._conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
            self._conn.commit()

    def get_profile(self, profile_id: int) -> dict[str, Any]:
        with self._lock:
            row = self._require_profile(profile_id)
            prompts = self._prompts_for(profile_id)
            return {
                "id": row["id"],
                "name": row["name"],
                "globals": row["globals"],
                "prompts": prompts,
            }

    def set_globals(self, profile_id: int, text: str) -> None:
        with self._lock:
            self._require_profile(profile_id)
            self._conn.execute(
                "UPDATE profiles SET globals = ? WHERE id = ?", (text, profile_id)
            )
            self._conn.commit()

    # ----- prompts --------------------------------------------------------

    def push_prompt(self, profile_id: int, text: str) -> dict[str, Any]:
        """Upsert ``text`` to the front of the history and enforce the cap.

        An existing entry keeps its ``fav`` flag and bumps to newest. The
        non-favourite tail is trimmed to :data:`NONFAV_CAP`. Returns the
        resulting prompt list (newest first).
        """
        with self._lock:
            self._require_profile(profile_id)
            now = time.time()
            existing = self._conn.execute(
                "SELECT fav FROM prompts WHERE profile_id = ? AND text = ?",
                (profile_id, text),
            ).fetchone()
            fav = existing["fav"] if existing else 0
            self._conn.execute(
                "INSERT INTO prompts (profile_id, text, fav, created_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(profile_id, text) DO UPDATE SET created_at = excluded.created_at",
                (profile_id, text, fav, now),
            )
            self._trim_nonfav(profile_id)
            self._conn.commit()
            return {"prompts": self._prompts_for(profile_id)}

    def fav_prompt(self, profile_id: int, text: str, fav: bool) -> None:
        with self._lock:
            self._require_profile(profile_id)
            self._conn.execute(
                "UPDATE prompts SET fav = ? WHERE profile_id = ? AND text = ?",
                (1 if fav else 0, profile_id, text),
            )
            self._conn.commit()

    def delete_prompt(self, profile_id: int, text: str) -> None:
        with self._lock:
            self._require_profile(profile_id)
            self._conn.execute(
                "DELETE FROM prompts WHERE profile_id = ? AND text = ?",
                (profile_id, text),
            )
            self._conn.commit()

    def clear_prompts(self, profile_id: int) -> dict[str, Any]:
        """Delete non-favourite prompts; favourites are kept."""
        with self._lock:
            self._require_profile(profile_id)
            cur = self._conn.execute(
                "DELETE FROM prompts WHERE profile_id = ? AND fav = 0", (profile_id,)
            )
            self._conn.commit()
            return {"removed": cur.rowcount}

    # ----- migration ------------------------------------------------------

    def import_profile(
        self, name: str, globals_text: str, prompts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Bulk-seed a profile from migrated ``localStorage`` data.

        ``prompts`` arrives in display order (newest/favs first). We assign
        decreasing timestamps so ``created_at DESC`` preserves that order.
        """
        name = _clean_name(name) if name.strip() else DEFAULT_PROFILE_NAME
        with self._lock:
            now = time.time()
            try:
                pid = self._insert_profile(name, globals_text, now)
            except sqlite3.IntegrityError:
                raise ValueError(f"a profile named {name!r} already exists")
            seen: set[str] = set()
            for i, p in enumerate(prompts):
                text = p.get("text")
                if not isinstance(text, str) or not text or text in seen:
                    continue
                seen.add(text)
                self._conn.execute(
                    "INSERT INTO prompts (profile_id, text, fav, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (pid, text, 1 if p.get("fav") else 0, now - i),
                )
            self._trim_nonfav(pid)
            self._conn.commit()
            return {"id": pid, "name": name, "prompt_count": len(seen)}

    # ----- internals ------------------------------------------------------

    def _insert_profile(self, name: str, globals_text: str, ts: float) -> int:
        cur = self._conn.execute(
            "INSERT INTO profiles (name, globals, created_at) VALUES (?, ?, ?)",
            (name, globals_text, ts),
        )
        return int(cur.lastrowid)

    def _require_profile(self, profile_id: int) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT id, name, globals FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise ProfileNotFound(f"no profile with id {profile_id}")
        return row

    def _prompts_for(self, profile_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT text, fav FROM prompts WHERE profile_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (profile_id,),
        ).fetchall()
        return [{"text": r["text"], "fav": bool(r["fav"])} for r in rows]

    def _trim_nonfav(self, profile_id: int) -> None:
        """Keep only the newest :data:`NONFAV_CAP` non-favourite prompts."""
        self._conn.execute(
            "DELETE FROM prompts WHERE id IN ("
            "  SELECT id FROM prompts WHERE profile_id = ? AND fav = 0 "
            "  ORDER BY created_at DESC, id DESC LIMIT -1 OFFSET ?"
            ")",
            (profile_id, NONFAV_CAP),
        )


_STORE: SqliteProfileStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> SqliteProfileStore:
    """Lazily open the process-wide store (after ensuring OUT_DIR exists)."""
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                ensure_dirs()
                _STORE = SqliteProfileStore(DB_PATH)
    return _STORE
