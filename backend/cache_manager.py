from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"
CACHE_FOLDER = DATA_FOLDER / "cache"

CACHE_ITEMS = {
    "extracted_text": DATA_FOLDER / "extracted_text",
    "analysis": DATA_FOLDER / "analysis",
    "summaries": DATA_FOLDER / "summaries",
    "flashcards": DATA_FOLDER / "flashcards",
    "quizzes": DATA_FOLDER / "quizzes",
    "podcasts": DATA_FOLDER / "podcasts",
}


def calculate_file_hash(file_path: Path) -> str:
    """Return a stable SHA-256 fingerprint for an uploaded file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_cache_key(
    file_hash: str,
    document_mode: str,
    podcast_length: str,
) -> str:
    """
    Build a cache key from the file and output-changing settings.
    """

    raw_key = (
        f"{file_hash}|{document_mode}|{podcast_length}"
    )

    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


def cache_path(cache_key: str) -> Path:
    return CACHE_FOLDER / cache_key


def metadata_path(cache_key: str) -> Path:
    return cache_path(cache_key) / "metadata.json"


def cache_exists(cache_key: str) -> bool:
    path = cache_path(cache_key)

    return (
        path.exists()
        and metadata_path(cache_key).exists()
    )


def remove_path_safely(
    path: Path,
    attempts: int = 3,
    delay_seconds: float = 0.4,
) -> None:
    """
    Remove a file or folder with short retries for Windows/OneDrive locks.

    If removal still fails, restoration can continue because copytree uses
    dirs_exist_ok=True and overwrites matching files.
    """

    if not path.exists():
        return

    for attempt in range(1, attempts + 1):
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)

            return

        except (PermissionError, OSError):
            if attempt == attempts:
                return

            time.sleep(delay_seconds * attempt)


def copy_folder_contents(
    source_folder: Path,
    target_folder: Path,
) -> None:
    """
    Copy a folder into an existing or new destination safely.

    dirs_exist_ok=True is essential on Windows because OneDrive may recreate
    a destination folder immediately after it has been removed.
    """

    target_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        source_folder,
        target_folder,
        dirs_exist_ok=True,
    )


def save_cache(
    *,
    cache_key: str,
    original_filename: str,
    file_hash: str,
    document_mode: str,
    podcast_length: str,
    generated_tasks: Iterable[str],
) -> Path:
    """Copy current generated resources into a cache entry."""

    destination = cache_path(cache_key)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_items: list[str] = []

    for item_name, source_folder in CACHE_ITEMS.items():
        if not source_folder.exists():
            continue

        target_folder = destination / item_name

        remove_path_safely(target_folder)
        copy_folder_contents(
            source_folder,
            target_folder,
        )

        saved_items.append(item_name)

    metadata = {
        "cache_key": cache_key,
        "original_filename": original_filename,
        "file_hash": file_hash,
        "document_mode": document_mode,
        "podcast_length": podcast_length,
        "generated_tasks": list(generated_tasks),
        "saved_items": saved_items,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    metadata_path(cache_key).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return destination


def restore_cache(cache_key: str) -> dict:
    """
    Restore cached resources into the live data folders.

    This version is safe when Windows or OneDrive recreates a destination
    directory between deletion and copying.
    """

    source = cache_path(cache_key)

    if not cache_exists(cache_key):
        raise FileNotFoundError(
            "The requested cache entry does not exist."
        )

    for item_name, target_folder in CACHE_ITEMS.items():
        cached_folder = source / item_name

        if not cached_folder.exists():
            continue

        remove_path_safely(target_folder)

        target_folder.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        copy_folder_contents(
            cached_folder,
            target_folder,
        )

    return json.loads(
        metadata_path(cache_key).read_text(
            encoding="utf-8",
            errors="replace",
        )
    )


def clear_cache_entry(cache_key: str) -> None:
    """Delete one cache entry without affecting the rest of the cache."""

    remove_path_safely(
        cache_path(cache_key)
    )
