"""Refresh the two HTTP datasets; experiment runner defaults to cached data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from emlab.data import load_material
for name in ['silica','water']:
    _,p=load_material(name,Path(__file__).resolve().parents[1]/'data/raw',refresh=True)
    print(name,p['sha256'])
