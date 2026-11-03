"""Point the scripts at wherever you unpacked this supplement and your raw datasets.

The scripts were written with absolute paths, which is honest about how they were run but
useless on another machine. This rewrites the two roots they use, in place, across every
file in experiments/.

    python set_paths.py                     # supplement root = this directory
    python set_paths.py --external "C:/data/prognostics"

The first root is this supplement, which already contains the derived arrays needed by most
scripts. The second is the directory holding the raw sets that are too large to ship:
C-MAPSS (train_FD001.txt, train_FD003.txt, CMAPSSData/), XJTU-SY bearing dataset/,
NASABattery/, phm-ieee-2012-data-challenge-dataset-master/. Scripts that need it are listed
in README.md; the rest run against the shipped arrays alone.

Re-running is safe: the replacement is idempotent as long as the new roots are given the
same way each time. Use --dry-run to see what would change.
"""
import argparse
import glob
import io
import os

BS = chr(92)
OLD_HERE = 'D:' + BS + 'claudecode' + BS + 'generationbound'
OLD_EXT = ('D:' + BS + 'b\u00e1ch khoa \u0110\u00e0 N\u1eb5ng' + BS + 'nghiencuukhoahoc'
           + BS + 'baotri' + BS + 'data')


def variants(p):
    """A path literal appears raw, with its backslashes escaped, and with forward slashes."""
    return [p, p.replace(BS, BS + BS), p.replace(BS, '/')]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--here', default=here,
                    help='root of this supplement (default: its own directory)')
    ap.add_argument('--external', default=None,
                    help='directory holding the raw datasets that are not shipped')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    new_here = os.path.abspath(a.here).rstrip(BS + '/')
    pairs = list(zip(variants(OLD_HERE), variants(new_here)))
    if a.external:
        new_ext = os.path.abspath(a.external).rstrip(BS + '/')
        pairs += list(zip(variants(OLD_EXT), variants(new_ext)))

    files = sorted(glob.glob(os.path.join(new_here, 'experiments', '*.py')))
    if not files:
        raise SystemExit(f'no scripts found under {os.path.join(new_here, "experiments")}')

    touched = 0
    for f in files:
        src = io.open(f, encoding='utf-8').read()
        out = src
        for old, new in pairs:
            out = out.replace(old, new)
        if out != src:
            touched += 1
            if not a.dry_run:
                io.open(f, 'w', encoding='utf-8').write(out)

    verb = 'would rewrite' if a.dry_run else 'rewrote'
    print(f'{verb} {touched} of {len(files)} scripts')
    print(f'  supplement root -> {new_here}')
    if a.external:
        print(f'  raw dataset root -> {os.path.abspath(a.external)}')
    else:
        print('  raw dataset root left as written; pass --external to set it')


if __name__ == '__main__':
    main()
