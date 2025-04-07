# Assignment 4: Survival Analysis

## Prerequisistes

- `numpy ~= 1.26`
- `lifelines ~= 0.30.0`
- `pandas ~= 2.2`
- `pandas = { version = ">=2.2.3,<3", extras = ["excel"] }`
- `scikit-survival ~= 0.24.1`
- `scikit-learn ~= 1.6.1`
- `seaborn ~= 0.13.2`
- `matplotlib ~= 3.10.0`
- `pathlib ~= 1.0.1`
- `openpyxl ~= 3.1.5`
- `category_encoders ~= 2.8.1`

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
pixi add scikit-survival
pixi add category_encoders
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
- https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://www.kaggle.com/code/prashant111/random-forest-classifier-tutorial
