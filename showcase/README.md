# LOFA showcase

This directory demonstrates STREAM on a loss-of-flow accident in a four-channel MTR-like loop, run with the changes on this branch.

## Notebooks

Read them in order:

1. `01-how-to-think-about-stream.ipynb` explains how Calculations combine in an Aggregator, then solves a small heated loop steady and in time, including the messages STREAM gives when a solve goes wrong.
2. `02-the-lofa.ipynb` follows the pump trip and scram through coastdown and flapper opening to natural circulation. It checks the results by hand, then shows the full-power run stopping when the hot channel reaches saturation.
3. `03-stress-test.ipynb` reads 94 runs of the case, each varying one parameter around the operating point, and sorts them into completed, saturation, failed and timeout.
4. `04-what-changed.ipynb` runs the LOFA benchmark ladder at every commit of the branch and reverts each later commit alone to show which ones the accident needs.

## Opening them

Inside the `stream-env` environment, run `jupyter lab showcase/` from the repository root. Jupyter starts each kernel in `showcase/`, and `pip install -e .` installs only `stream`, so the notebooks need the repository root on the path; the first cell of each notebook puts it there, and no PYTHONPATH is needed. Without any environment, open the files in `showcase/html/` in a browser; that directory holds HTML exports of the executed notebooks.

## Re-running a sweep

Each sweep axis writes one CSV to `results/`. To re-run one, run `python -m showcase.sweeps --axis power --workers 8` from the repository root (`--axis all` runs every axis), or set `RECOMPUTE = True` in notebook 3.

## Environment

```
conda env create -f allreq.yml -n stream-env
conda install -n stream-env -y jupyterlab
conda activate stream-env
pip install -e .
```

## Results

`results/` holds the records produced on 2026-10-05 on this branch, the 94 sweep cases plus the two commit tables, and the notebooks render from them.
