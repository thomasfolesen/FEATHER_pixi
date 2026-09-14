# FEATHER with Pixi

[![Arxiv](https://img.shields.io/badge/ArXiv-2005.07959-orange.svg)](https://arxiv.org/abs/2005.07959)

This repository is a reproducibility-oriented version of the original
[FEATHER repository](https://github.com/benedekrozemberczki/FEATHER)
by Benedek Rozemberczki and Rik Sarkar.

It contains the Python reference implementation of **FEATHER** and
**FEATHER-G** from the CIKM '20 paper:

> **Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models**

The main additions in this repository are:

- a **Pixi-managed environment**
- support for both **Linux (`linux-64`)** and **Windows (`win-64`)**
- a committed `pixi.lock` file for reproducibility
- a small fix to `src/main.py` so the FEATHER command-line parameters
  `--order`, `--eval-points`, and `--theta-max` are actually passed to the
  FEATHER model

The goal is to preserve the original implementation as much as possible while
making it easy for collaborators on Linux and Windows to reproduce the same
environment.

<p align="center">
  <img width="600" src="charfun.jpg">
</p>

---

## Original paper

FEATHER was introduced in:

> **Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models**  
> Benedek Rozemberczki and Rik Sarkar  
> CIKM, 2020

Paper:

https://arxiv.org/abs/2005.07959

Original repository:

https://github.com/benedekrozemberczki/FEATHER

The datasets are also available through SNAP:

http://snap.stanford.edu/

FEATHER is also available in the Karate Club package:

https://github.com/benedekrozemberczki/karateclub

---

## Table of Contents

1. [Reproducibility setup](#reproducibility-setup)
2. [Installation with Pixi](#installation-with-pixi)
3. [Requirements](#requirements)
4. [Input](#input)
5. [Options](#options)
6. [Examples](#examples)
7. [Changes from the original repository](#changes-from-the-original-repository)
8. [Citing](#citing)
9. [License](#license)

---

## Reproducibility setup

The original FEATHER README states that the code was developed using Python
3.5.2 and the following package versions:

```text
networkx          2.4
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
```

For this repository, the environment has been reproduced using:

```text
Python            3.6.15
networkx          2.4
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
```

`argparse` is not installed separately because it is part of the Python
standard library.

The environment is managed with **Pixi**.

The project targets:

```text
linux-64
win-64
```

This allows Linux and Windows collaborators to use the same project definition
and lock file.

---

## Installation with Pixi

First install Pixi:

https://pixi.sh/

Then clone this repository:

```sh
git clone git@github.com:thomasfolesen/FEATHER_pixi.git
cd FEATHER_pixi
```

Install the locked environment:

```sh
pixi install
```

Check the Python version:

```sh
pixi run python --version
```

The expected version is:

```text
Python 3.6.15
```

You can also verify the main package versions with:

```sh
pixi run python -c "import numpy, pandas, scipy, networkx, tqdm, texttable; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('scipy', scipy.__version__); print('networkx', networkx.__version__); print('tqdm', tqdm.__version__); print('texttable', texttable.__version__)"
```

Expected versions:

```text
numpy 1.15.4
pandas 0.23.4
scipy 1.1.0
networkx 2.4
tqdm 4.28.1
texttable 1.5.0
```

### Why Python 3.6?

The original code was written around Python 3.5.2.

For this reproducibility setup, Python 3.6.15 is used because it allows the old
scientific Python stack used by FEATHER to be resolved for both Linux and
Windows while remaining close to the original development environment.

Pixi is the authoritative environment manager for this repository. There is no
need to create a separate `venv`.

---

## Requirements

The exact environment is described by:

```text
pixi.toml
pixi.lock
```

The Pixi workspace includes both Linux and Windows:

```toml
platforms = ["linux-64", "win-64"]
```

The scientific dependencies are intentionally kept close to the versions used
by the original FEATHER implementation.

This repository follows the principle:

> **Reproduce first, improve second.**

The old dependencies are therefore intentional and should not be upgraded
without first checking that FEATHER produces equivalent results.

---

## Input

### Node level

The code takes an input graph stored as a CSV edge list.

Each row represents an edge between two nodes separated by a comma.

The first row is a header.

Nodes should be indexed consecutively starting from `0`.

The default edge file is:

```text
input/edges/ER_edges.csv
```

The feature matrix is dense and is also stored as CSV.

The default feature file is:

```text
input/features/ER_features.csv
```

An example feature matrix looks like:

| Feature 1 | Feature 2 | Feature 3 | Feature 4 |
| --- | --- | --- | --- |
| 3 | 0 | 1.37 | 1 |
| 1 | 1 | 2.54 | -11 |
| 2 | 0 | 1.08 | -12 |
| 1 | 1 | 1.22 | -4 |
| ... | ... | ... | ... |
| 5 | 0 | 2.47 | 21 |

### Graph level

Graphs are stored in a JSON file where keys are graph identifiers and values
are edge lists.

Graph identifiers should be consecutive and start at `0`.

Nodes inside each graph should also be indexed starting from `0`.

The implementation assumes that the graphs are connected.

Example:

```json
{
  "0": [[0, 1], [1, 2], [2, 3]],
  "1": [[0, 1], [1, 2], [2, 0]],
  "2": [[0, 1], [1, 2]]
}
```

The default graph-level input is:

```text
input/graphs/ER_graphs.json
```

---

## Options

Learning the embedding is handled by:

```text
src/main.py
```

To see the command-line interface:

```sh
pixi run python src/main.py --help
```

### Input and output options

```text
--graph-input      STR   Input edge list CSV.
                        Default: input/edges/ER_edges.csv

--feature-input    STR   Input feature CSV.
                        Default: input/features/ER_features.csv

--graphs-input     STR   Input graph collection JSON.
                        Default: input/graphs/ER_graphs.json

--output           STR   Embedding output path.
                        Default: output/ER_node_embedding.csv
```

### Model options

```text
--model-type       STR     FEATHER or FEATHER-G.
                          Default: FEATHER

--eval-points      INT     Number of characteristic-function evaluation points.
                          Default: 25

--order            INT     Random-walk / adjacency propagation order.
                          Default: 5

--theta-max        FLOAT   Maximum characteristic-function evaluation point.
                          Default: 2.5
```

---

## Examples

### Run the default FEATHER example

```sh
pixi run python src/main.py
```

This writes to:

```text
output/ER_node_embedding.csv
```

Be aware that this will overwrite the existing file if one already exists.

### Run a small FEATHER test

For a quick smoke test:

```sh
pixi run python src/main.py \
  --order 1 \
  --eval-points 2 \
  --output ./output/small_test.csv
```

This uses:

```text
order = 1
eval_points = 2
```

and writes the result to:

```text
output/small_test.csv
```

This is useful for checking that the environment, input loading, FEATHER
calculation, and output writing all work without running the full default
configuration.

### Change the random-walk order

```sh
pixi run python src/main.py \
  --order 3 \
  --output ./output/order_3_embedding.csv
```

### Change the number of evaluation points

```sh
pixi run python src/main.py \
  --eval-points 10 \
  --output ./output/eval_10_embedding.csv
```

### Change theta maximum

```sh
pixi run python src/main.py \
  --theta-max 5.0 \
  --output ./output/theta_5_embedding.csv
```

### Run FEATHER-G

```sh
pixi run python src/main.py \
  --model-type FEATHER-G \
  --output ./output/ER_graph_embedding.csv
```

---

## Understanding the node-level output

The output CSV contains:

```text
id,x_0,x_1,x_2,...
```

The `id` column contains the node identifier.

The remaining columns form the FEATHER node embedding.

For every input feature, FEATHER evaluates the real and imaginary parts of the
characteristic function using cosine and sine terms at multiple evaluation
points.

The embedding dimensionality is determined by:

```text
number of input features
× number of evaluation points
× 2
× order
```

where the factor of `2` corresponds to the cosine and sine components.

For example, with:

```text
16 input features
2 evaluation points
order = 1
```

the embedding contains:

```text
16 × 2 × 2 × 1 = 64
```

embedding dimensions:

```text
x_0 ... x_63
```

plus the separate `id` column.

---

## Changes from the original repository

### Pixi environment

The main addition is a Pixi environment that can be reproduced on both Linux
and Windows.

The repository includes:

```text
pixi.toml
pixi.lock
```

The lock file should normally be committed to Git so collaborators use the same
resolved dependency environment.

### FEATHER CLI parameter fix

The original `src/main.py` parsed the arguments:

```text
--theta-max
--eval-points
--order
```

but constructed the node-level model using:

```python
model = FEATHER()
```

This meant that the command-line values were not passed to the model and the
defaults defined by `FEATHER.__init__` were always used.

This repository changes that code to:

```python
model = FEATHER(
    theta_max=args.theta_max,
    eval_points=args.eval_points,
    order=args.order
)
```

As a result, commands such as:

```sh
pixi run python src/main.py --order 1 --eval-points 2
```

now actually use the requested values.

At present, this fix applies to the node-level `FEATHER` model.

The graph-level `FEATHERG` path has otherwise been kept unchanged from the
original implementation.

---

## Development workflow

For normal development:

```sh
git pull
```

Make changes on a branch when appropriate:

```sh
git switch -c my-feature
```

Run Python commands through Pixi:

```sh
pixi run python ...
```

This avoids accidentally using the system Python instead of the project
environment.

For example, on Linux the system may use a modern Python such as Python 3.12,
while FEATHER runs inside its isolated Pixi-managed Python 3.6 environment.

---

## Citing

If you use FEATHER in research, please cite the original paper:

```bibtex
@inproceedings{feather,
    title={Characteristic Functions on Graphs: Birds of a Feather,
           from Statistical Descriptors to Parametric Models},
    author={Benedek Rozemberczki and Rik Sarkar},
    year={2020},
    pages={1325--1334},
    booktitle={Proceedings of the 29th ACM International Conference
               on Information and Knowledge Management (CIKM '20)},
    organization={ACM}
}
```

---

## Attribution

The FEATHER algorithm and original reference implementation were created by:

- Benedek Rozemberczki
- Rik Sarkar

Original repository:

https://github.com/benedekrozemberczki/FEATHER

This repository primarily adds environment reproducibility and a small
command-line parameter fix for collaborative use.

---

## License

The original FEATHER repository is released under the MIT License.

See:

```text
LICENSE
```

Original license:

https://github.com/benedekrozemberczki/FEATHER/blob/master/LICENSE