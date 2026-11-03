"""Fetch the Zhang et al. (2020) impedance-and-capacity dataset from Zenodo.

Twelve LiCoO2/graphite coin cells cycled to end of life, with electrochemical impedance
spectra recorded every even cycle alongside the capacity record.  This is the only public set
found that carries two physically distinct channels on the same units at a usable fleet size:
capacity measures the bulk damage, impedance senses the interfaces.
"""
import urllib.request, os, zipfile, hashlib

DEST = r'D:\claudecode\generationbound\data\zhang'
FILES = {
    'EIS data.zip': 'https://zenodo.org/api/records/3633835/files/EIS%20data.zip/content',
    'Capacity data.zip': 'https://zenodo.org/api/records/3633835/files/Capacity%20data.zip/content',
}

os.makedirs(DEST, exist_ok=True)
for name, url in FILES.items():
    out = os.path.join(DEST, name)
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        print(f'{name}: already present ({os.path.getsize(out):,d} B)')
    else:
        print(f'fetching {name} ...', flush=True)
        urllib.request.urlretrieve(url, out)
        print(f'  {os.path.getsize(out):,d} B', flush=True)
    d = os.path.join(DEST, name[:-4].replace(' ', '_'))
    if not os.path.isdir(d):
        with zipfile.ZipFile(out) as z:
            z.extractall(d)
    n = sum(len(f) for _, _, f in os.walk(d))
    print(f'  extracted to {os.path.basename(d)}: {n} files')

for root, dirs, files in os.walk(DEST):
    if files:
        rel = os.path.relpath(root, DEST)
        print(f'  {rel}: {len(files)} files, e.g. {sorted(files)[:4]}')
