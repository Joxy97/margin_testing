"""Check the publication's byte inventory and original-file preservation."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parents[1]
inventory=json.loads((root/'SHA256SUMS.json').read_bytes())
for relative,expected in inventory.items():
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('File checksum mismatch: '+relative)
originals=json.loads((root/'reproducibility/original_file_preservation.json').read_bytes())
for record in originals:
    if not any(inventory.get(p)==record['sha256'] for p in record['current_paths']):
        raise SystemExit('Original file missing: '+record['original_path'])
print(f"Verified {len(inventory)} publication files and all {len(originals)} original files.")
