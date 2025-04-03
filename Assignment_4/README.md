# Assignment 4: Survival Analysis

## Prerequisistes

- `numpy ~= 1.26`
- `lifelines ~= 0.30.0`
- `pandas ~= 2.2`
- `scikit-survival ~= 0.24.1`

## How to run

To run `*/Scripts/main.ipynb`, run the following commands:

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
pixi add lifelines
pixi add pathlib
pixi add openpyxl
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

## Useful links
# For pixi
- https://pixi.sh/latest/basic_usage/
# For models
- https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html
- 
- 
# For Data sets
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
# For Visualization
- https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html