# BINF5507 - Assignment 2

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
# For ML models
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# For metrics
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
# For visualizations
https://seaborn.pydata.org/generated/seaborn.heatmap.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html


