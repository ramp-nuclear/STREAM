"""Fetch or install the MathJax files hosted alongside the documentation.

MathJax 4 ships its fonts as a separate npm package that conda-forge does not carry, so
both the core and the font are downloaded here as plain data, pinned (with integrity hashes)
by ``mathjax/package-lock.json``. That lockfile is also what Dependabot updates. npm is not
needed.

Online (CI, or a networked machine)::

    python mathjax_assets.py fetch
    python mathjax_assets.py fetch --bundle mathjax.tar.gz   # also pack for an air-gapped machine

Offline (air-gapped machine)::

    python mathjax_assets.py install mathjax.tar.gz

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
LOCKFILE = HERE / "mathjax" / "package-lock.json"
DEFAULT_DEST = HERE / "source" / "_static" / "mathjax"
VERSIONS_FILE = "MATHJAX_VERSIONS.json"

# Where each package goes relative to the destination, and which of its top-level entries to
# keep (None keeps everything). The font's cjs/, mjs/ and svg/ trees are for Node or SVG output
# and are not needed by the CHTML output the docs use.
LAYOUT = {
    "mathjax": (".", None),
    "@mathjax/mathjax-newcm-font": ("fonts/mathjax-newcm-font", {"chtml", "chtml.js", "package.json", "LICENSE"}),
}


def locked_packages() -> dict[str, dict]:
    """The pinned packages from the lockfile, keyed by package name."""
    lock = json.loads(LOCKFILE.read_text())
    packages = {}
    for key, info in lock["packages"].items():
        name = key.removeprefix("node_modules/")
        if name in LAYOUT:
            packages[name] = {"version": info["version"], "resolved": info["resolved"], "integrity": info["integrity"]}
    missing = LAYOUT.keys() - packages.keys()
    if missing:
        sys.exit(f"{LOCKFILE} is missing {sorted(missing)}")
    return packages


def verify_integrity(data: bytes, integrity: str, name: str) -> None:
    algorithm, _, expected = integrity.partition("-")
    actual = base64.b64encode(hashlib.new(algorithm, data).digest()).decode()
    if actual != expected:
        sys.exit(f"Integrity check failed for {name}: expected {integrity}, got {algorithm}-{actual}")


def safe_extractall(archive: tarfile.TarFile, dest: Path, members=None) -> None:
    if hasattr(tarfile, "data_filter"):
        archive.extractall(dest, members=members, filter="data")
        return
    for member in members if members is not None else archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
            sys.exit(f"Refusing to extract unsafe archive member {member.name!r}")
    archive.extractall(dest, members=members)


def unpack_package(data: bytes, dest: Path, keep: set[str] | None) -> None:
    """Extract an npm tarball, dropping its leading ``package/`` directory."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = []
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts[1:]
            if not parts or (keep is not None and parts[0] not in keep):
                continue
            member.name = str(PurePosixPath(*parts))
            members.append(member)
        safe_extractall(archive, dest, members)


def replace_dir(staged: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), dest)


def fetch(dest: Path, bundle: Path | None) -> None:
    packages = locked_packages()
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "mathjax"
        staged.mkdir()
        for name, info in packages.items():
            print(f"Downloading {name}@{info['version']}")
            with urllib.request.urlopen(info["resolved"], timeout=60) as response:
                data = response.read()
            verify_integrity(data, info["integrity"], name)
            subdir, keep = LAYOUT[name]
            unpack_package(data, staged / subdir, keep)
        (staged / VERSIONS_FILE).write_text(json.dumps(packages, indent=2, sort_keys=True) + "\n")
        replace_dir(staged, dest)
    print(f"MathJax installed in {dest}")
    if bundle is not None:
        with tarfile.open(bundle, "w:gz") as archive:
            archive.add(dest, arcname="mathjax")
        print(f"Bundle written to {bundle}")


def install(bundle: Path, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(bundle, "r:gz") as archive:
        safe_extractall(archive, Path(tmp))
        staged = Path(tmp) / "mathjax"
        if not (staged / VERSIONS_FILE).is_file():
            sys.exit(f"{bundle} is not a MathJax bundle made by 'fetch --bundle'")
        installed = json.loads((staged / VERSIONS_FILE).read_text())
        replace_dir(staged, dest)
    print(f"MathJax installed in {dest}")
    expected = locked_packages()
    if installed != expected:
        print(
            f"WARNING: the bundle does not match {LOCKFILE.name}:\n"
            f"  bundle:   { {name: info['version'] for name, info in installed.items()} }\n"
            f"  lockfile: { {name: info['version'] for name, info in expected.items()} }\n"
            "Make a new bundle with 'fetch --bundle' from the same revision of the repository.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_parser = commands.add_parser("fetch", help="download the pinned MathJax files (needs network)")
    fetch_parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    fetch_parser.add_argument("--bundle", type=Path, help="also write a .tar.gz bundle for 'install'")

    install_parser = commands.add_parser("install", help="install a bundle made by 'fetch --bundle' (offline)")
    install_parser.add_argument("bundle", type=Path)
    install_parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)

    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.dest, args.bundle)
    else:
        install(args.bundle, args.dest)


if __name__ == "__main__":
    main()
