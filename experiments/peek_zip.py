"""List the contents of a remote zip without downloading it.

The CFRP archive is 4.6 GB, almost all of it raw Lamb-wave waveforms, and what decides whether
it is usable is the number of specimens, not the waveform volume.  A zip keeps its central
directory at the end of the file, so a couple of range requests are enough to read the file
list and count the panels before committing to the download.
"""
import urllib.request, struct, io, zipfile, sys, re

URL = sys.argv[1] if len(sys.argv) > 1 else \
    'https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip'
UA = {'User-Agent': 'Mozilla/5.0'}


def get(url, start, end):
    h = dict(UA)
    h['Range'] = f'bytes={start}-{end}'
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180).read()


size = int(urllib.request.urlopen(
    urllib.request.Request(URL, headers=UA), timeout=120).headers['Content-Length'])
print(f'archive {size:,d} bytes')

tail = get(URL, size - 200000, size - 1)
i = tail.rfind(b'PK\x06\x06')                       # zip64 end of central directory
if i >= 0:
    cd_size, cd_off = struct.unpack('<QQ', tail[i+40:i+56])
else:
    j = tail.rfind(b'PK\x05\x06')
    cd_size, cd_off = struct.unpack('<II', tail[j+12:j+20])
print(f'central directory: {cd_size:,d} bytes at offset {cd_off:,d}')

cd = get(URL, cd_off, cd_off + cd_size - 1)
names, pos = [], 0
while pos < len(cd) - 4 and cd[pos:pos+4] == b'PK\x01\x02':
    nlen, elen, clen = struct.unpack('<HHH', cd[pos+28:pos+34])
    usz = struct.unpack('<I', cd[pos+24:pos+28])[0]
    nm = cd[pos+46:pos+46+nlen].decode('utf-8', 'ignore')
    names.append((nm, usz))
    pos += 46 + nlen + elen + clen

print(f'{len(names)} entries')
tops = sorted({n.split('/')[0] for n, _ in names})
print(f'top level: {tops[:20]}')
subs = sorted({'/'.join(n.split('/')[:2]) for n, _ in names if n.count('/') >= 1})
print(f'{len(subs)} second-level entries, first 30:')
for s in subs[:30]:
    print('   ' + s)
