# PatternCode

PatternCode is a Python tool for design of optimal labeling patterns for optical genome mapping via information theory.
<!-- See the [paper]() for details. -->

# Installation

```shell
conda env create -n patterncode -f ./environment.yml
conda activate patterncode
```

# Download genome data from NCBI
```shell
python ./patterncode/genome_data.py download_genomes
```

# Reproduce the figures from the paper
Run the [jupyter notebook](./paper_figures.ipynb)
