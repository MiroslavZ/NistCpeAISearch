"""Lossless report input, durable checkpoints and process-wide API key locks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import BinaryIO


INPUT_FIELDS = ("input_vendor", "input_product", "input_version")
RESULT_FIELDS = ("cpe_part", "cpe_vendor", "cpe_product", "status", "conclusion")


@dataclass(frozen=True)
class InputRow:
    input_vendor: str
    input_product: str
    input_version: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.input_vendor, self.input_product, self.input_version


def read_report(path: str | Path) -> list[InputRow]:
    """Read only inventory fields; old mappings never enter the matching context."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        names = reader.fieldnames or []
        missing = set(INPUT_FIELDS) - set(names)
        if missing:
            raise ValueError("В CSV отсутствуют столбцы: " + ", ".join(sorted(missing)))
        if any(names.count(name) != 1 for name in INPUT_FIELDS):
            raise ValueError("Входные столбцы CSV не должны повторяться")
        rows = []
        for record in reader:
            if any(record[name] is None for name in INPUT_FIELDS):
                raise ValueError(f"Неполная запись CSV около строки {reader.line_num}")
            rows.append(InputRow(*(record[name] for name in INPUT_FIELDS)))
        return rows


class State:
    """One committed result per exact input triple, bound to input and settings."""

    def __init__(
        self, path: str | Path, input_digest: str, config_digest: str,
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.input_digest = input_digest
        self.config_digest = config_digest
        self.resume = resume
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> State:
        exists = self.path.exists()
        if exists and not self.resume:
            raise ValueError("Файл состояния уже существует; используйте --resume или новый путь")
        if not exists and self.resume:
            raise ValueError("Файл состояния для продолжения не найден")
        self._connection = sqlite3.connect(self.path)
        try:
            # FULL sync ensures a committed row survives interruption of the run.
            self._connection.execute("PRAGMA synchronous=FULL")
            expected = {
                "schema_version": "1", "input_digest": self.input_digest,
                "config_digest": self.config_digest,
            }
            if exists:
                actual = dict(self._connection.execute("SELECT name, value FROM metadata"))
                if actual != expected:
                    raise ValueError("Состояние несовместимо с исходным CSV или конфигурацией")
            else:
                with self._connection:
                    self._connection.execute(
                        "CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
                    )
                    self._connection.executemany(
                        "INSERT INTO metadata VALUES (?, ?)", expected.items()
                    )
                    self._connection.execute(
                        "CREATE TABLE results ("
                        "vendor TEXT NOT NULL, product TEXT NOT NULL, version TEXT NOT NULL, "
                        "payload TEXT NOT NULL, status TEXT NOT NULL, "
                        "PRIMARY KEY (vendor, product, version))"
                    )
            return self
        except BaseException:
            self._connection.close()
            self._connection = None
            raise

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Состояние должно быть открыто через контекстный менеджер")
        return self._connection

    def get(self, row: InputRow) -> dict | None:
        result = self.connection.execute(
            "SELECT payload FROM results WHERE vendor=? AND product=? AND version=?", row.key
        ).fetchone()
        return json.loads(result[0]) if result else None

    def save(self, row: InputRow, result: dict) -> None:
        payload = {field: result[field] for field in RESULT_FIELDS}
        if not all(isinstance(value, str) for value in payload.values()):
            raise ValueError("Все поля результата должны быть строками")
        with self.connection:
            self.connection.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(vendor, product, version) DO UPDATE "
                "SET payload=excluded.payload, status=excluded.status",
                (*row.key, json.dumps(payload, ensure_ascii=False), payload["status"]),
            )

    def reset_statuses(self, statuses: set[str]) -> None:
        """Forget only requested statuses so the next run retries these triples."""
        with self.connection:
            self.connection.executemany(
                "DELETE FROM results WHERE status=?", ((status,) for status in statuses)
            )

    def __exit__(self, *exc: object) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def export_report(path: str | Path, rows: list[InputRow], state: State) -> None:
    """Atomically replace the report, including duplicates and unfinished rows."""
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(*INPUT_FIELDS, *RESULT_FIELDS))
            writer.writeheader()
            for row in rows:
                result = state.get(row)
                if result is None:
                    result = dict.fromkeys(RESULT_FIELDS, "")
                    result.update(status="search_incomplete", conclusion="Обработка записи не завершена.")
                writer.writerow(dict(zip(INPUT_FIELDS, row.key)) | result)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RunLock:
    """Nonblocking OS locks; crashes release locks without deleting lock files.

    Each key is locked separately, so sharing either API key blocks another run.
    Only SHA-256 digests appear in filenames; key values are never written out.
    """

    def __init__(self, keys: list[str]) -> None:
        self._digests = sorted({hashlib.sha256(key.encode()).hexdigest() for key in keys})
        self._handles: list[BinaryIO] = []

    def __enter__(self) -> RunLock:
        directory = Path(tempfile.gettempdir()) / "cpe-matcher-locks"
        directory.mkdir(mode=0o700, exist_ok=True)
        try:
            for digest in self._digests:
                handle = (directory / f"{digest}.lock").open("a+b")
                try:
                    if os.name == "nt":
                        import msvcrt
                        # Windows byte-range locks require an existing byte.
                        if os.fstat(handle.fileno()).st_size == 0:
                            handle.write(b"0")
                            handle.flush()
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    handle.close()
                    raise RuntimeError("Другой запуск уже использует один из API-ключей") from error
                self._handles.append(handle)
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *exc: object) -> None:
        for handle in reversed(self._handles):
            # Closing the descriptor releases its OS lock, even after a crash.
            handle.close()
        self._handles.clear()
