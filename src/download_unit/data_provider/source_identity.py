"""Content revisions for configured local sources, refreshed on metadata changes."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path


@lru_cache(maxsize=256)
def _fileDigest(path: str, size: int, modified: int, changed: int) -> str:
    digest = sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def localSourceRevision(parameters) -> tuple[tuple[str, str], ...]:
    locations = parameters.get("locations", (parameters.get("location", ""),))
    revisions = []
    for location in locations:
        if "://" in str(location):
            revisions.append((str(location), str(parameters.get("revision", ""))))
            continue
        path = Path(location).expanduser().resolve()
        stat = path.stat()
        revisions.append((str(path), _fileDigest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)))
    return tuple(revisions)
