"""Restore all pinned official SCAN text files, with per-file hash checks."""
import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import hashlib

ROOT=Path(__file__).resolve().parent


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--archive',help='Use a local codeload zip instead of downloading')
 args=parser.parse_args(argv)
 manifest=json.loads((ROOT/'source_manifest.json').read_text())
 raw=Path(args.archive).read_bytes() if args.archive else urllib.request.urlopen(
  'https://codeload.github.com/brendenlake/SCAN/zip/'+manifest['revision'],timeout=120).read()
 archive=zipfile.ZipFile(io.BytesIO(raw))
 expected={r['source_path']:r for r in manifest['files']};found={}
 for info in archive.infolist():
  relative='/'.join(info.filename.split('/')[1:])
  if relative in expected:
   data=archive.read(info)
   if hashlib.sha256(data).hexdigest()!=expected[relative]['sha256']:
    raise ValueError('Official data hash mismatch: '+relative)
   found[relative]=data
 if set(found)!=set(expected):raise ValueError('Incomplete SCAN archive')
 (ROOT/'raw').mkdir(exist_ok=True)
 for relative,data in found.items():
  dest=ROOT/'raw'/expected[relative]['local_file']
  if dest.exists() and dest.read_bytes()!=data:raise ValueError('Existing data differs: '+str(dest))
  dest.write_bytes(data)
 print(f'Restored {len(found)} official SCAN text files, 104 paired splits.')


if __name__=='__main__':main()
