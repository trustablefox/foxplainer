[![Total alerts](https://img.shields.io/lgtm/alerts/g/trustablefox/foxplainer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/trustablefox/foxplainer/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/trustablefox/foxplainer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/trustablefox/foxplainer/context:python)
[![codecov](https://codecov.io/gh/trustablefox/foxplainer/branch/main/graph/badge.svg?token=SvHpnT1Sey)](https://codecov.io/gh/trustablefox/foxplainer)[![Documentation Status](https://readthedocs.org/projects/foxplainer/badge/?version=latest)](https://foxplainer.readthedocs.io/en/latest/?badge=latest)

<p align="center">
<img src="img/new_logo.png" width="100" height="100">
 
<div align="center">
<h1>
    <b>
     FoX: a Fo rmal eX plainer for JIT Defect Prediction
    </b>
</h1>
</div>

</p>

![FoX demo](./img/fox_demo.gif)

## Experiment Replication
**To replicate our experiment in the paper, please go to [this repository](https://github.com/trustablefox/exp_replication).**

## FoX 

```bibtex
@underreview{
}
```

## FoX Example

```bash
git clone https://github.com/trustablefox/quickstart_example.git
cd quickstart_example
```
Then, open the [DEMO.ipynb](https://github.com/trustablefox/quickstart_example/blob/main/DEMO.ipynb) file for a concrete example.

## Quick Start FoX Tool

```bash
pip3 install foxplainer
```

```python
from foxplainer.explainer import FoX
fx = FoX(global_model_name="LR", 
           xnum='all', 
           global_model_path="./global_model/openstack_LR_global_model.pkl", 
           proj_name="openstack", 
           data_path="./dataset/",
           inst_id=5)
fx.explain(in_jupyter=False)
```

A user needs to provide the following parameters:
1. **global_model_name**: should be either ***"RF"*** (Random Forest) or ***"LR"*** (Logistic Regression)
2. **xtype (optional)**: Only needed when **xnum=1**. Support two types of explanations, specify ***"abd"*** for Abductive Explanation or ***"con"*** for Contrastive Explanation
3. **xnum**: should be either 1 or "all" (return all explanations)
4. **global_model_path**: the path to your trained global model (model should be trained using sci-kit learn library)
5. **proj_name**: the project name of your dataset
6. **data_path**: the path to the required data, **this path should contain the following 2 files**:
   
   6.a ***{proj_name}.csv*** - the complete file consisting of all training + testing data with features and label
   
   6.b ***{proj_name}_X_test.csv*** - the testing data that only contains feature columns without label  

7. **inst_id**: the row number of the instance to be explained in your testing data

## Table of Contents

* **[Installation](#installation-fox-tool)**
  * [Install via pip](#install-via-pip)
  * [Install via conda](#install-via-conda)
  * [Local install via poetry](#local-install-via-poetry)

* **[Contributions](#contributions)**

* **[Documentation](#documentation)**

* **[License](#license)**

## Install FoX Tool
```bash
pip3 install -r requirements.txt
```

### Install via pip
```bash
pip3 install foxplainer
```

### Install via conda
```bash
conda install -c conda-forge foxplainer
```

### Local install via poetry
```bash
git clone https://github.com/trustablefox/foxplainer.git
poetry install
```
If you do not have poetry installed yet, simply run the command below to install it, for more information, click [here](https://python-poetry.org/docs/).
```bash
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
```

## Contributions

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/trustablefox/foxplainer/graphs/contributors).

## Documentation  
Access our official documentation [here](https://foxplainer.readthedocs.io/en/latest/?).

## License
[MIT License](https://github.com/trustablefox/foxplainer/blob/main/LICENSE)

