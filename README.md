# Baseline Predictive Pipeline -- ETAI

20231645 - João Patrício

## Runs

| Week | Model | Train Accuracy | Test Accuracy |
|----|----|-------|----|
| 2 | LR | 0.680 | 0.680 |
| 2 | DT | 0.829 | 0.628 |
| 3 | LR | 0.676 | 0.657 |
| 3 | DT | 0.792 | 0.612 |

| Week | Model | Train Accuracy (mean) | Val Accuracy (mean) | OOF Accuracy|
|----|----|-------|----|----|
| 5 | Dummy | 0.549 | 0.549 | 0.55 |
| 5 | LR | 0.675 | 0.672 | 0.67 |
| 5 | DT | 0.695 | 0.610 | 0.619 |
| 5 | DT (tuning) | 0.685 | 0.672 | 0.67 |
| 5 | RF | 0.733 | 0.650 | 0.65 |
| 5 | RF (tuning) | 0.733 | 0.650 | 0.68 |
| 6 | Adaboost | 0.680 | 0.674 | 0.67 |
| 6 | ORF| 0.781 | 0.645 | 0.65 |


### Week 2 

Decision tree is clearly overfitting. So far, Logistic Regression is the better option.

### Week 3

We actually made the model worse with the preprocessing? Decision Tree still overfitting.

### Week 5

Every model above dummy's baseline.
Tuning prevents overfitting in decision tree. 
DT and LR improved from recent weeks
RF added
Tuning is positive.

### Week 6

ORF overfits.
Adaboost may be the best untuned model. 

### Week 8



## Intro

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py            # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   ├── results.py          # saves each run's report to disk
|   └── tuning.py           # hyperparameter tuning with optuna
├── results/                # created automatically -- one file per run (not tracked in git)
├── praticals/              # pratical classes notebooks 
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Setting up the Github repository. Initial runs with default Logistic Regression and Decision Tree. | Personalization of the README file. Some changes in the config.yaml file |
| 3 | Preprocessing | First runs. Preprocessing. |
| 4 | Cross Validation | Incomplete work |
| 5 | Hyperparameter tuning | Bringing repository up to date. Results updated|
| 6 | Oblique Random Forests and Adaboost. | Added Oblique Random Forest and Adaboost. Added 2 new features.|
| 7 | Boosting and Metrics | Added 3 boosting models to the pipeline. |
| 8 | Feature Selection | Updated Metrics. |


