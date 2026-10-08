#!/usr/bin/env python3
"""Execute a collector with vendored project imports and isolated artifact paths."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys
from runtime_env import ROOT,RUNTIME,environment
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--python',default=sys.executable)
p.add_argument('--artifacts',type=Path,default=ROOT/'artifacts')
p.add_argument('script',type=Path)
p.add_argument('arguments',nargs=argparse.REMAINDER)
a=p.parse_args()
script=a.script.resolve()
if not script.is_relative_to(ROOT): p.error('Collector must belong to this release')
raise SystemExit(subprocess.call([a.python,str(script),*a.arguments],cwd=RUNTIME,env=environment(a.artifacts)))
