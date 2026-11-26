# pySUTO

pySUTO (Python-based Supply-Use Table Optimiser) reconciles supply-use tables from
conflicting sources. Each source datum carries an uncertainty; a sigma-weighted quadratic
programme, solved with Clarabel, returns the flows that agree with the accounting
constraints at the smallest weighted distance from the data, and the posterior uncertainty
of every flow. It is built for material flow analysis and was developed on copper.

```
pip install pysuto
pip install "pysuto[tutorial]"
pysuto tutorial DEST
```

The second line adds Jupyter and Plotly; `pysuto tutorial DEST` copies the tutorial
notebooks into the directory `DEST`. Without a Chrome browser, run `plotly_get_chrome -y`
once: Kaleido needs it to write the Sankey figures. Python 3.9 or newer is required.

`python -m pysuto run` requires `--case` and `--data-dir`. `--case` is a bundled alias
(`toy`, `toy_coproduct`), a dotted module or a file path, each naming a module that declares
a `CASE`; the package ships no study case. The copper chain runs only from a repository
checkout (`--case scripts/copper.py`) with the licensed original data.

pySUTO is licensed under the MIT License. Documentation, the method, the citation file and
the source are in the GitHub repository:
https://github.com/florihu/pySUTO

To report a bug or to contribute a change, read the contributing guide:
https://github.com/florihu/pySUTO/blob/main/CONTRIBUTING.md
