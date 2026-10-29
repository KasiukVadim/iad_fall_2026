"""HTTP material data with a checked-in cache and SHA-256 provenance."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
import requests,yaml
SOURCES={'silica':('SiO2','Malitson'),'water':('H2O','Daimon-20.0C')}
BASE='https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main'

def load_material(name,root,refresh=False):
    mat,page=SOURCES[name];p=Path(root)/f'{mat}_{page}.yml'
    url=f'{BASE}/{mat}/nk/{page}.yml'
    if refresh or not p.exists():
        r=requests.get(url,timeout=60);r.raise_for_status();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
    b=p.read_bytes();d=yaml.safe_load(b)
    return d,{'name':name,'url':url,'sha256':hashlib.sha256(b).hexdigest(),
              'references':d['REFERENCES'],'retrieved_at_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),
              'kind':'published dispersion fit, not raw field measurements'}

def refractive_index(doc,wavelength_um):
    d=doc['DATA'][0];c=np.array([float(v) for v in d['coefficients'].split()]);w=np.asarray(wavelength_um)
    lo,hi=map(float,d['wavelength_range'].split())
    if np.any((w<lo)|(w>hi)): raise ValueError('Wavelength outside source validity range')
    n2=1+c[0]+np.zeros_like(w,dtype=float)
    power={'formula 1':2,'formula 2':1}[d['type']]
    for a,b in zip(c[1::2],c[2::2]):n2+=a*w*w/(w*w-b**power)
    return np.sqrt(n2)
