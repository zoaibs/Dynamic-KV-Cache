"""Extract official CPython include/libs only; no installers or system changes."""
import zipfile
from scripts.resources import ROOT,download

def main():
    download('https://api.nuget.org/v3-flatcontainer/python/3.11.9/python.3.11.9.nupkg',
             '.runtime/python-dev.nupkg')
    root=(ROOT/'.runtime/gpu').resolve()
    with zipfile.ZipFile(ROOT/'.runtime/python-dev.nupkg') as z:
        for name in z.namelist():
            if name.startswith(('tools/include/','tools/libs/')) and not name.endswith('/'):
                target=(root/name.removeprefix('tools/')).resolve()
                if not target.is_relative_to(root):raise RuntimeError('Unsafe archive member')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))

if __name__=='__main__':main()
