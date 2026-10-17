# Building the documentation

Create the environment from `requirements.yml`, install STREAM (`pip install -e .` from the repository root),
and fetch MathJax, which is hosted with the docs instead of loaded from a CDN:

```sh
cd docs
python mathjax_assets.py fetch
make html
```

The MathJax version is pinned in `mathjax/package-lock.json`, which Dependabot keeps up to date.
`mathjax_assets.py` only reads it, so npm is not needed.

## Building without network access

On a machine with network access, from the same revision of the repository:

```sh
python docs/mathjax_assets.py fetch --bundle mathjax.tar.gz
```

Carry `mathjax.tar.gz` over, then on the offline machine:

```sh
cd docs
python mathjax_assets.py install /path/to/mathjax.tar.gz
make html
```

`install` warns if the bundle was made from a different MathJax pin than the checked-out lockfile.
