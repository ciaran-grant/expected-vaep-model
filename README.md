expected-vaep-model
==============================

Building, exploring and visualising an Expected Valuing Actions by Estimating Probabilities model.

Valuing Actions by Estimating Probabilities (VAEP)
----------------------------------------------------
VAEP is a framework to assign each action on the pitch a value and overcomes some of the flaws in Expected Threat.
- Values all actions (Handballs, Kicks, Marks, Shots, Ground Ball Gets etc.)
- Introduces game context in addition to only locations
- Considers offensive and defensive consequences of actions

VAEP values are calculated by training two models based on current game context (previous 3 actions gamestate information):
- Probability of scoring (in next 10 actions)
- Probability of conceding (in next 10 actions)

Assuming every action is made to maximise scoring proabilites and minimise conceding probabilities, combining these two probabilities will give an overall net value to every action.

Similar to Expected Threat, the VAEP values are calculated from the scoring and conceding probabilities before/after the action takes place.

Eg. A pass back towards your own goal could decrease your probability of scoring and increase your probability of conceding in the next few actions. 

|        | Scoring % | Conceding % |
| ------ | :-------: | :---------: |
|  Pre   | 10%       | 1%          |
| Post   | 5%        | 4%          |

|        | Offensive |  Defensive  | VAEP |
| ------ | :-------: | :---------: | :--: |
| Value  | -5%       | +3%         |  -8% |


Every action will have an offensive value, defensive value and an overall VAEP value.

[VAEP: socceraction](https://socceraction.readthedocs.io/en/latest/index.html)

[Exploring VAEP](https://dtai.cs.kuleuven.be/sports/vaep)

[Explaining how VAEP works](https://dtai.cs.kuleuven.be/sports/blog/exploring-how-vaep-values-actions/)

Tom Decroos, Lotte Bransen, Jan Van Haaren, and Jesse Davis. “Actions speak louder than goals: Valuing player actions in soccer.” In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1851-1861. 2019.

Maaike Van Roy, Pieter Robberechts, Tom Decroos, and Jesse Davis. “Valuing on-the-ball actions in soccer: a critical comparison of xT and VAEP.” In Proceedings of the AAAI-20 Workshop on Artifical Intelligence in Team Sports. AI in Team Sports Organising Committee, 2020.

Expected VAEP
-------------
Expected VAEP is an extension to the above VAEP approach which attempts to overcome a single challenge that I forsaw with VAEP that I will try to explain below.

Scores (Goals + Behinds) are the basis on what AFL is scored on and the ultimate aim of every game is to score more than the opponent, maximise your scores and minimise your opponents scores. The issue with using Scores is that they are volatile in predicting future performance, sometimes a shot goes in and sometimes a shot misses.

Incomes xScore, which quantifies the expected value of each shot based on the likelihood of the shot being a goal (or behind) based on the location and game context. (Similar to xG in soccer.) The advantage here is that you can quantify how well a team performs based on the shots they manufacture (process) rather than the scores those shots produce (outcome). Expected Score and Expected Goals are actually more predictive of future Scores/Goals than Scores/Goals themselves are.

In the same fashion as Scores (Goals + Behinds), VAEP still relies on there being a goal scored in the next 10 actions to quantify the value. The same action will produce a goal sometimes and will produce a miss (or not shot at all) sometimes, whilst the action still produces the same quality of shot (same xScore). This means that the weight or importance of that action is de-valued due to the volatility of Scores.

So in the same veign as Scores being measured by the quality of shots with xScores, I have abstracted out valueing actions by measuring the quality of shots in the next 10 shots. So technically I have replaced the binary Goal/No Goal in the next 10 actions response from VAEP with the continous xScore value of the shot in the next 10 actions (0 if no shots). This means that actions further down the chain are not influenced by the result of shots at the end of the chain.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
