"""Reproduce two complete exact searches for the authentic public weighted graph.

Run only in the isolated reference directory, not in a benchmark campaign. The
output is reference evidence and is never supplied to benchmark solver calls.
No search timeout, step cutoff or invented/reweighted input is permitted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time

INPUT_SHA256 = '68963a69f2c5423e6a4ab0860027e7a1c2c908dce40eb00c81f1fb3ccb2f505e'
AUTHOR_COMMIT = 'f156fbf6b3fb1779e9efc274043f76b46c241898'
CLIQUER_ARCHIVE_SHA256 = 'ff306d27eda82383c0257065e3ffab028415ac9af73bccfdd9c2405b797ed1f1'


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def command(args, directory):
    started = time.perf_counter()
    result = subprocess.run(args, cwd=directory, capture_output=True, text=True, check=False)
    record = dict(command=args, exit_code=result.returncode,
                  elapsed_s=time.perf_counter()-started, stdout=result.stdout,
                  stderr=result.stderr, completed=True, configured_time_limit=None,
                  configured_step_limit=None, timed_out=False)
    if result.returncode != 0:
        raise RuntimeError(json.dumps(record))
    return record


def run(directory, source_path, output_path):
    directory = Path(directory).resolve()
    source_path = Path(source_path).resolve()
    if fingerprint(source_path) != INPUT_SHA256:
        raise ValueError('The original public weighted graph checksum differs')
    commit = command(['git', 'rev-parse', 'HEAD'], directory/'author-source')['stdout'].strip()
    if commit != AUTHOR_COMMIT or command(['git', 'status', '--porcelain'], directory/'author-source')['stdout'].strip():
        raise ValueError('Author source must be pinned and unmodified')
    archive = directory/'cliquer-1.21.tar.gz'
    if fingerprint(archive) != CLIQUER_ARCHIVE_SHA256:
        raise ValueError('Official Cliquer source archive checksum differs')
    # Independently verify all C/header/Makefile sources against that archive.
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            if member.isfile() and (member.name.endswith(('.c', '.h', '/Makefile'))):
                if hashlib.sha256(stream.extractfile(member).read()).hexdigest() != fingerprint(directory/member.name):
                    raise ValueError('Cliquer source differs from its official release')
    sanitizers = ['-fsanitize=signed-integer-overflow', '-fno-sanitize-recover=all']
    builds = [command(['g++', '-O3', '-DNDEBUG', '-std=c++23', *sanitizers,
                       '-I', 'author-source/lib', 'verify_continuous_reference.cpp', '-o', 'verify_author'], directory),
              command(['make', '-B', '-C', 'cliquer-1.21',
                       'CFLAGS=-O3 -Wall '+' '.join(sanitizers),
                       'LDFLAGS=-fsanitize=signed-integer-overflow', 'cl'], directory)]
    author = command([str(directory/'verify_author'), str(source_path)], directory)
    author_result = json.loads(author['stdout'])
    if not author_result.get('completed') or not author_result.get('exact') or author_result['configured_step_limit'] is not None:
        raise ValueError('Authors exact search did not complete without a cutoff')
    cliquer = command([str(directory/'cliquer-1.21/cl'), '-q', '-q', '-s', '-m', '0', '-M', '0', str(source_path)], directory)
    match = re.fullmatch(r'\s*size=(\d+), weight=(\d+):\s*([\d\s]+)', cliquer['stdout'])
    if not match:
        raise ValueError('Unexpected Cliquer global-maximum output')
    labels = list(map(int, match[3].split()))
    if len(labels) != int(match[1]) or int(match[2]) != author_result['weight_integer']:
        raise ValueError('The two complete exact searches disagree')
    if author['stderr'].strip() or cliquer['stderr'].strip():
        raise ValueError('Search sanitizer/error output is not clean')
    author.update(method='authors_MCQDW', exact=True, weight_integer=author_result['weight_integer'],
                  source_vertex_labels=author_result['source_vertex_labels'],
                  executable_sha256=fingerprint(directory/'verify_author'),
                  source_commit=commit, source_tag='mcqdw_v1.0.0',
                  source_url='https://gitlab.com/janezkonc/insidrug/-/releases/mcqdw_v1.0.0',
                  source_archive_url='https://gitlab.com/janezkonc/insidrug/-/archive/mcqdw_v1.0.0/insidrug-mcqdw_v1.0.0.tar.gz',
                  source_archive_sha256='bd33e3ec96b4f31c9e931adedc685150752f51187ab6afdb0f69637ba7dbd5a5')
    cliquer.update(method='independent_Cliquer', exact=True, weight_integer=int(match[2]),
                   source_vertex_labels=labels, executable_sha256=fingerprint(directory/'cliquer-1.21/cl'),
                   source_url='https://users.aalto.fi/~pat/cliquer.html', source_version='1.21',
                   source_archive_url='https://users.aalto.fi/~pat/cliquer/cliquer-1.21.tar.gz',
                   source_archive_sha256=CLIQUER_ARCHIVE_SHA256)
    proof = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                 status='independently_verified_optimum', published_numeric_reference=False,
                 input_sha256=INPUT_SHA256, original_vertex_weights_preserved=True,
                 completed_exact_searches=[author, cliquer], builds=builds,
                 driver_sha256=fingerprint(directory/'verify_continuous_reference.cpp'),
                 verification_tool_sha256=fingerprint(Path(__file__)),
                 compiler=command(['g++', '--version'], directory)['stdout'],
                 objective_max=author_result['weight_integer'],
                 objective_min=-author_result['weight_integer'],
                 formal_optimality_certificate_available=False,
                 assertion_note='Authors CMake uses -DNDEBUG. The equality of recursively shrinking candidate capacity and global degree-vector size is a wrong debug invariant; global original vertex-ID bounds remain valid. Source is unmodified.',
                 proof_note='Two independently implemented complete exact searches agree; feasible witnesses will also be scored independently against original source and canonical QUBO. This is not a published optimum or a separate formal proof certificate.')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(proof, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=proof['status'], objective_min=proof['objective_min'],
                          completed_exact_searches=2, proof_sha256=fingerprint(output_path))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.run_dir, args.source, args.output)


if __name__ == '__main__':
    main()
