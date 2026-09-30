"""Atomic metadata and URL-addressed, hash-verified source cache."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import ssl
import certifi
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor
import re


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def writeJson(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def readJson(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rejectHtml(payload, contentType=''):
    prefix = payload[:2048].lstrip().lower()
    if not payload or 'html' in contentType.lower() or b'<html' in prefix or b'<!doctype html' in prefix:
        raise ValueError('Expected benchmark data; received empty content or an HTML/error/login page')
    if b'\x00' in prefix:
        raise ValueError('Expected text data; received binary content')


class SourceCache:
    """A recorded SHA is a local fingerprint, not an authenticity certificate."""

    def __init__(self, directory, kind='raw'):
        self.directory = Path(directory) / kind
        self.directory.mkdir(parents=True, exist_ok=True)

    def paths(self, url):
        key = sha256(url.encode('utf-8'))
        return self.directory / (key+'.raw'), self.directory / (key+'.json')

    def read(self, url):
        raw, metadata = self.paths(url)
        record = readJson(metadata)
        payload = raw.read_bytes()
        if record['source_url'] != url or record['sha256'] != sha256(payload) or record['bytes'] != len(payload):
            raise ValueError(f'Cached source integrity failure: {url}')
        return payload, record

    def fetch(self, url, offline=False, timeout=30., retries=2, allowDocument=False):
        raw, metadata = self.paths(url)
        if raw.exists() or metadata.exists():
            return self.read(url)
        if offline:
            raise FileNotFoundError(f'Offline cache missing: {url}')
        if urlsplit(url).scheme not in ('https', 'http'):
            raise ValueError('Only public HTTP(S) source URLs are accepted')
        error = None
        for attempt in range(retries+1):
            try:
                request = Request(url, headers={'User-Agent': 'QUBOBenchmark/1.0 (research data download)'})
                with urlopen(request, timeout=timeout, context=ssl.create_default_context(cafile=certifi.where())) as response:
                    payload = response.read(64*1024*1024+1)
                    declaredLength = response.headers.get('Content-Length')
                    if declaredLength is not None and len(payload) != int(declaredLength):
                        raise ValueError(f'Truncated response: expected {declaredLength} bytes, received {len(payload)}')
                    contentType = response.headers.get('Content-Type', '')
                    resolved = response.geturl()
                if len(payload) > 64*1024*1024:
                    raise ValueError('Source exceeds the 64 MiB download limit')
                if not allowDocument:
                    rejectHtml(payload, contentType)
                elif not payload:
                    raise ValueError('Empty reference document')
                record = dict(source_url=url, resolved_url=resolved, retrieved_at=stamp(),
                              sha256=sha256(payload), bytes=len(payload), content_type=contentType,
                              hash_meaning='local download fingerprint, not independent authenticity proof')
                temporary = raw.with_suffix('.tmp')
                temporary.write_bytes(payload)
                temporary.replace(raw)
                writeJson(metadata, record)
                return payload, record
            except Exception as exc:
                error = exc
                if attempt < retries:
                    time.sleep(min(2., .25*2**attempt))
        raise RuntimeError(f'Download failed after {retries+1} attempts: {url}: {error}') from error

    def recoverRanges(self, url, retrievalUrl, validate, chunkSize=8000, workers=6):
        """Explicit recovery of a truncated source, never a silent source substitution.

        Require one stable ETag, exact Content-Range/length and successful format
        validation before replacing the rejected response. No remote code runs.
        """
        context=ssl.create_default_context(cafile=certifi.where())
        def request(start,end):
            for attempt in range(3):
                try:
                    req=Request(retrievalUrl,headers={'Range':f'bytes={start}-{end}','User-Agent':'QUBOBenchmark/1.0'})
                    with urlopen(req,timeout=12,context=context) as response:
                        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                        if response.status != 206 or not match:raise ValueError('Publisher did not honor byte range')
                        begin,finish,total=map(int,match.groups())
                        if begin != start or finish != min(end,total-1):raise ValueError('Unexpected content range')
                        payload=response.read(finish-begin+1)
                        if len(payload) != finish-begin+1:raise ValueError('Truncated range')
                        return payload,total,response.headers.get('ETag')
                except Exception:
                    if attempt == 2:raise
        first,total,etag=request(0,chunkSize-1)
        if total > 64*1024*1024:raise ValueError('Source exceeds size limit')
        if not etag:raise ValueError('Range recovery requires a stable publisher ETag')
        def part(start):
            payload,size,tag=request(start,min(start+chunkSize-1,total-1))
            if size != total or tag != etag:raise ValueError('Source changed during range retrieval')
            return payload
        pool=ThreadPoolExecutor(max_workers=workers)
        futures=[pool.submit(part,start) for start in range(chunkSize,total,chunkSize)]
        try:payload=first+b''.join(f.result() for f in futures)
        finally:pool.shutdown(wait=True,cancel_futures=True)
        validate(payload)
        raw,meta=self.paths(url)
        if raw.exists():
            rejected=self.directory.parent/'rejected_downloads';rejected.mkdir(exist_ok=True)
            (rejected/raw.name).write_bytes(raw.read_bytes())
            if meta.exists():(rejected/meta.name).write_bytes(meta.read_bytes())
        record=dict(source_url=url,resolved_url=retrievalUrl,retrieved_at=stamp(),sha256=sha256(payload),bytes=len(payload),
                    content_type='text/plain',etag=etag,range_chunk_bytes=chunkSize,
                    retrieval_note='Explicit same-publisher HTTP byte-range recovery after HTTPS truncation; exact ranges and complete format validated.',
                    hash_meaning='local fingerprint, not independent authenticity proof')
        temporary=raw.with_suffix('.tmp');temporary.write_bytes(payload);temporary.replace(raw);writeJson(meta,record)
        return payload,record
