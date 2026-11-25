<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figs/logo/pysuto-logo-dark.png">
    <img src="figs/logo/pysuto-logo-light.png" alt="pySUTO: Python-based Supply–Use Table Optimiser" width="600">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/pysuto/"><img src="https://img.shields.io/pypi/v/pysuto" alt="PyPI version"></a>
</p>

> ⭐ If pySUTO is useful to your work, please leave a star on [GitHub](https://github.com/florihu/pySUTO) — it helps others find the project.
>
> 🏷️ Filing an [issue](https://github.com/florihu/pySUTO/issues/new)? Please give it a label — `bug`, `enhancement`, `documentation` or `question` — so it lands in the right queue; the maintainer then sets its triage state (`needs-triage`, `ready-for-agent`, …).

A modular Python framework for constructing, reconciling, and propagating uncertainty through material flow accounts derived from heterogeneous data sources. pySUTO takes raw statistical data feeds, maps them onto a common accounting structure, aggregates across sources with inverse-variance weighting, imputes missing values, builds a constrained linear system, and solves a σ-weighted QP — tracking uncertainty end-to-end from raw pedigree scores to the final optimum.

> Contributors / LLM assistants: see [CLAUDE.md](CLAUDE.md) for behavioural rules.


---

## Install

```bash
pip install pysuto
pip install "pysuto[tutorial]"     # adds Jupyter, Plotly and Kaleido, for the notebooks
plotly_get_chrome -y               # once: Kaleido needs Chrome to write the Sankey PNGs
```

Python 3.9 or newer is required. The core dependencies are `numpy`, `pandas`, `scipy`, `pyarrow`, `clarabel`, `pyyaml`, `openpyxl` and `matplotlib` (only [diag/](src/pysuto/diag/) plots with it).

**From a checkout** (development, the copper application, the tests):

```bash
git clone https://github.com/florihu/pySUTO.git
cd pySUTO
pip install -e ".[dev]"
python -m pytest tests/            # about 5 min; in a git worktree, prefix PYTHONPATH=src
```

Bit-identity of the stored arrays is claimed in the environment pinned by [environment-reference.yml](environment-reference.yml) (linux-64): `conda env create -f environment-reference.yml && conda activate pysuto-ref`. [environment.yml](environment.yml) builds a development environment on any platform.

---

## Quickstarts

**1. The tutorial.** Eleven notebooks run the six-stage pipeline on a small synthetic toy case in about 1.5 h. The last chapter writes a case of your own outside the package and runs it. The chapter list is in [tutorial/README.md](src/pysuto/tutorial/README.md).

```bash
pysuto tutorial pysuto-tutorial    # copies the notebooks and the params template
cd pysuto-tutorial && jupyter lab 00_world.ipynb
```

**2. The toy chain.** One command runs all six stages on the toy case. It needs no data and takes about 25 s. Each stage runs in a fresh child process.

```bash
python -m pysuto run --case toy --data-dir toy-data
```

`--case` takes a bundled alias (`toy`, `toy_coproduct`), a dotted module on `sys.path` or a file path (`./my_case.py`); each names a module that declares a `CASE`. `--from` and `--to` select a slice of the chain (`feed`, `imp`, `const`, `opt`, `uncprop`, `psut`). The run trees appear under `toy-data/proc/<stage>/<run_id>/`; the logs are in `toy-data/logs/`. pySUTO ships no study case (ADR 0047).

**3. The copper chain.** This runs only from a checkout with the licensed source data under `data/input/raw` and the input tables that carry no licence yet (see *Licence and citation*). The repository holds neither.

```bash
python scripts/acc.py --id copper
# by hand, once: the BACI parse, the gross registration of BACI and icsg_clean.py --to-feed
python -m pysuto run --case scripts/copper.py --data-dir data
```

The by-hand steps and their order are in [docs/architecture.md](docs/architecture.md) (*Stage 1*). Keep 3 GB of disk free: one constraint run takes about 0.85 GB. The solve takes about 2 min on 8 cores; start the chain in the background.

---

## How it works

Six stages, each with its own driver, its own run ids and its own persisted output. The mathematics of each stage is in [docs/architecture.md](docs/architecture.md).

| Stage | Does | Persists |
|---|---|---|
| 1 Feed | Maps each raw source onto the common account with its pedigree-based σ. | `feed_…` |
| 2 Aggregate and impute | Pools the feeds by inverse variance and fills the edges no feed covers. | `imp_…` |
| 3 Constraints | Builds the sparse linear system (mass balance, sums, ratios, bounds). | `const_…` |
| 4 Solver | Solves one σ-weighted QP per period with Clarabel. | `opt_…` |
| 5 Uncertainty | Propagates the posterior σ of every flow (closed form). | `uncprop_…` |
| 6 PSUT | Writes the solved flows as a tidy supply-use table; Sankeys and a derived input-output view. | `psut_…` |

Every artefact carries a value and its σ. Hard rows hold exactly; soft rows are priced by σ. A sweep re-runs Stages 2–4 once per knob point. The twelve commitments that no case can change are listed in [docs/architecture.md](docs/architecture.md#what-the-framework-assumes).

---

## Where things are

| What | Where |
|---|---|
| Library | [src/pysuto/](src/pysuto/): the six stages, the registries, the solver, [diag/](src/pysuto/diag/) (diagnostics). It knows no study case. |
| Application (copper) | [scripts/](scripts/): the case, the feeds and builders, the stage drivers, the sweeps, the reports. See [scripts/README.md](scripts/README.md). |
| Tooling | [tools/](tools/): input status, issue audit, run pruning, tutorial refresh. |
| Tests | [tests/](tests/); `tests/copper/` holds the tests that need the copper application. Run them with `python -m pytest tests/` (never bare `pytest`). |
| Tutorial | [src/pysuto/tutorial/](src/pysuto/tutorial/): the toy case and the eleven notebooks. |
| Architecture and modules | [docs/architecture.md](docs/architecture.md): the stages, their mathematics, the module reference, the import direction. |
| Workflow | [docs/workflow.md](docs/workflow.md): the driver walk-through, run ids, sidecars, input registries, knobs, sweeps. |
| Vocabulary | [GLOSSARY.md](GLOSSARY.md). |
| Method and results | [docs/manuscript/](docs/manuscript/): `method.md`, `si.md` and the results. |
| Records | [docs/adr/](docs/adr/) (decisions), [docs/rep/](docs/rep/) (analyses of runs), [docs/history/](docs/history/) (closed rounds), [scripts/explo/](scripts/explo/) (the scripts behind the records). |
| Contributors | [CLAUDE.md](CLAUDE.md) (rules), [CODING_STANDARDS.md](CODING_STANDARDS.md), [docs/agents/](docs/agents/), [CHANGELOG.md](CHANGELOG.md). |

---

## How to cite

To cite pySUTO, use [CITATION.cff](CITATION.cff). GitHub shows it under "Cite this repository".

**Licence.** The code is licensed under the MIT License ([LICENSE](LICENSE)). The hand-authored tables `data/input/acc/`, `data/input/params/{conc,eff}/` and `data/input/mask/` are licensed under CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)). The other input tables (the feed concordances, metadata and weights, and the parameter tables `icsg_aggregates`, `ore_grade`, `scrap_share`, `stock_bounds` and `stripping_ratio`) carry no licence until their licence check is recorded (#297), so the repository does not hold them; the original source data are not in the repository and stay under their owners' terms. The optional `uncprop` extra pulls in `scikit-sparse`, whose CHOLMOD backend is LGPL/GPL; pySUTO does not need it.

---

## Links

- Source and issues: <https://github.com/florihu/pySUTO>
- Package: <https://pypi.org/project/pysuto/>
- Architecture: [docs/architecture.md](docs/architecture.md) · Workflow: [docs/workflow.md](docs/workflow.md) · Glossary: [GLOSSARY.md](GLOSSARY.md)
- References (Laner et al. 2016; Lenzen et al. 2009; Pauliuk et al. 2019): [docs/architecture.md](docs/architecture.md#references)
