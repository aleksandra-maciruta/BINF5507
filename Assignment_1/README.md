# BINF5507 - Assignment 1

## How to run

To run `Scripts/main.ipynb`, run the following commands:

```shell
brew install pixi
pixi
pixi init
pixi add jupiter
pixi add ipykernel
pixi add pandas
pixi add numpy
pixi add seaborn
pixi add matplotlib
pixi add scipy
pixi add scikit-learn
pixi run python -m ipykernel install --user --name=pixi-env --display-name "Python (Pixi)"
```

<!-- ```shell
brew install pixi
pixi
pixi init
pixi add jupiter
pixi add ipykernel
pixi add pandas
pixi add numpy
pixi add seaborn
pixi add matplotlib
pixi add scipy
pixi add scikit-learn
pixi run python -m ipykernel install --user --name=pixi-env --display-name "Python (Pixi)"
``` -->

Run tests with the following command:
```shell
pixi run python ./Assignment_1/Scripts/run_tests.py
```

## Useful links
# For pixi
- https://pixi.sh/latest/basic_usage/
# For visualizations
- https://matplotlib.org/stable/api/pyplot_summary.html
- https://seaborn.pydata.org/generated/seaborn.pairplot.html
# For redundancy removal
- https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python 