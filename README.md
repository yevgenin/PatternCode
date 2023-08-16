# PatternCode

PatternCode is a Python tool for design of optimal labeling patterns for optical genome mapping via information theory.
See the [paper](https://www.biorxiv.org/content/10.1101/2023.05.23.541882v1) for details.

# Reproducing the figures from the paper

Assuming anaconda python 3.10 is installed:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ./paper_figures.ipynb
```

# Usage example in jupyter notebook: [example.ipynb](./example.ipynb)

```shell
jupyter notebook ./example.ipynb
```
