"""Assemble a self-contained experiment source tree from shared versioned files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent

def assemble(variant, output):
    catalog = json.loads((ROOT / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    if variant not in catalog['variants']:
        raise ValueError('Unknown variant: ' + variant)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    files = catalog['variants'][variant]['files']
    planned = []
    for logical, row in files.items():
        relative = PurePosixPath(logical)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid logical path')
        source = (ROOT / row['stored']).resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError('Source path leaves the code directory')
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != row['sha256']:
            raise ValueError('Source SHA mismatch: ' + row['stored'])
        target = output / logical
        if not target.resolve().is_relative_to(output):
            raise ValueError('Output path leaves the requested directory')
        planned.append((target, content))
    for target, content in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    report = {'variant': variant, 'files': files}
    (output / 'assembled_sources.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return len(planned)

def main():
    catalog = json.loads((ROOT / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=sorted(catalog['variants']), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    count = assemble(args.variant, args.out)
    print(json.dumps({'variant': args.variant, 'files': count, 'output': str(args.out)}))

if __name__ == '__main__':
    main()
