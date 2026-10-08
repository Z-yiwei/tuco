#!/usr/bin/env python3
"""Pack and restore experiment assets listed in configs/artifacts.json."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def entries(manifest, groups=None):
    result = [x for x in manifest['files'] if not groups or x['group'] in groups]
    if not result: raise ValueError('No artifacts match requested groups')
    seen=set()
    for item in result:
        path=PurePosixPath(item['path'])
        if path.is_absolute() or '..' in path.parts or str(path) in seen: raise ValueError('Invalid/duplicate artifact path')
        seen.add(str(path))
    return result

def verify(root, files):
    errors=[]
    for item in files:
        p=root/item['path']
        if not p.is_file(): errors.append(f"missing: {item['path']}")
        elif p.is_symlink() or p.stat().st_size!=item['bytes'] or sha256(p)!=item['sha256']:
            errors.append(f"checksum/size mismatch: {item['path']}")
    if errors: raise ValueError('\n'.join(errors))
    return len(files)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','pack','restore'])
    parser.add_argument('--root',type=Path,default=ROOT/'artifacts')
    parser.add_argument('--manifest',type=Path,default=ROOT/'configs/artifacts.json')
    parser.add_argument('--group',action='append')
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text()); files=entries(manifest,args.group)
    root=args.root.resolve()
    if args.command=='verify':
        print(f'verified {verify(root,files)} artifacts'); return
    if not args.archive: parser.error('--archive is required')
    if args.command=='pack':
        verify(root,files)
        if args.archive.exists(): raise FileExistsError(args.archive)
        args.archive.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(args.archive.parent).free < sum(x['bytes'] for x in files) + 64*1024*1024:
            raise OSError('Insufficient free space for a conservative archive-size bound')
        temporary = args.archive.with_name(args.archive.name + '.partial')
        if temporary.exists(): raise FileExistsError(temporary)
        try:
            with tarfile.open(temporary,'w:gz',compresslevel=1) as archive:
                for item in files: archive.add(root/item['path'],arcname=item['path'],recursive=False)
            os.replace(temporary, args.archive)
        finally:
            if temporary.exists(): temporary.unlink()
        receipt={'archive':args.archive.name,'bytes':args.archive.stat().st_size,'sha256':sha256(args.archive),'files':len(files),'groups':sorted({x['group'] for x in files})}
        args.archive.with_suffix(args.archive.suffix+'.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt)); return
    expected={x['path']:x for x in files}
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restore-',dir=root.parent) as tmp:
        staging=Path(tmp)
        with tarfile.open(args.archive,'r:*') as archive:
            seen=set()
            for member in archive:
                if member.name not in expected or member.name in seen or not member.isfile():
                    raise ValueError(f'Unexpected archive member: {member.name}')
                if member.size!=expected[member.name]['bytes']: raise ValueError(f'Wrong size: {member.name}')
                seen.add(member.name)
                target=staging/member.name;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as source,target.open('wb') as output: shutil.copyfileobj(source,output)
        verify(staging,files)
        # Check every existing destination before committing any file.
        for item in files:
            dest=root/item['path']
            if not dest.resolve().is_relative_to(root): raise ValueError('Destination escapes artifact root')
            if dest.exists(): verify(root,[item])
        for item in files:
            dest=root/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists(): os.replace(staging/item['path'],dest)
    print(f'restored and verified {len(files)} artifacts')

if __name__=='__main__': main()
