<p align="center">
<img src="img/logo.png" width="100" height="100">
 
<div align="center">
<h1>
    <b>
     FoX: Trustable Just-In-Time Explanations
    </b>
</h1>
</div>

</p>

![FxExplainer demo](./img/fx_demo.gif)

# FoX 

```bibtex
@underreview{
}
```

## Quick Start FoX Tool

```bash
cd fox
```

```python
from explainer import FoX
fx = FoX(global_model_name="LR", 
           xnum='all', 
           global_model_path="./global_model/openstack_LR_global_model.pkl", 
           proj_name="openstack", 
           data_path="./dataset/",
           inst_id=5)
fx.explain(in_jupyter=True)
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

Please check out [this demo notebook](https://github.com/foxplainer/foxplainer/blob/main/fox/DEMO.ipynb) as a concrete example.

## Table of Contents
* **[Replication of Experiments](#replication-of-experiments)**
  * [Install Prerequisites](#install-prerequisites)
  * [Replication of Enumeration Explanations 0](#replication-of-enumeration-explanations-0)
  * [Replication of Enumeration Explanations 1](#replication-of-enumeration-explanations-1)
  * [Parse Logs](#parse-logs)
  * [Replicate RQ1 (Correctness)](#replicate-rq1-correctness)
  * [Replicate RQ2 (Robustness)](#replicate-rq2-robustness)
  * [Replicate RQ3 (Time)](#replicate-rq3-runtime)

* **[Installation](#installation-fox-tool)**
  * [Install via pip](#install-via-pip)
  * [Install via conda](#install-via-conda)
  * [Local install via poetry](#local-install-via-poetry)

* **[Contributions](#contributions)**

* **[Documentation](#documentation)**

* **[License](#license)**

## Replication of Experiments

### Install Prerequisites
To install the required packages, please run:
```bash
pip3 install -r requirements.txt
```

### Train LR and RF Models for Openstack and Qt
```bash
cd exp_replication/src
./experiment/experiment_train.sh
```

### Replication of Enumeration Explanations 0
First time to run the set of experiments. All logs are saved in *logs*. Note that the experiments take a while
```bash
cd exp_replication/src
./experiment/experiment_fox_0.sh && ./experiment/experiment_other_0.sh
```

### Replication of Enumeration Explanations 1
Second time to run the set of experiments. All logs are saved in *exp_replication/logs*. Note that the experiments take a while
```bash
cd exp_replication/src
./experiment/experiment_fox_1.sh && ./experiment/experiment_other_1.sh
```

### Parse logs
Parsing the logs. Explanations in each log are stored in a json file in *exp_replication/expls*
```bash
cd exp_replication/src
./experiment/experiment_plogs.sh
```

### Replicate RQ1 Correctness
The results are saved in *exp_replication/res/csv/rq1_correctness.csv*. Note that this will take a while
```bash
cd exp_replication/src
./experiment/experiment_rq1.sh
```

### Replicate RQ2 Robustness
The results are saved in *exp_replication/res/csv/rq2_robust.csv*. Note that this will take a while
```bash
cd exp_replication/src
./experiment/experiment_rq2.sh
```

### Replicate RQ3 Runtime
The results are saved in *exp_replication/res/csv/rq3_runtime.csv*. Note that this will take a while
```bash
cd exp_replication/src
./experiment/experiment_rq3.sh
```

For more information of training and explaining each model, please click [here](https://github.com/foxplainer/foxplainer/tree/main/exp_replication/src).

## Install FoX Tool

### Install via pip
```bash
Will publish to PyPI and update after double-blind review
```

### Install via conda
```bash
Will publish to Anaconda and update after double-blind review
```

### Local install via poetry
Will update after double-blind review

## Contributions

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/foxplainer/foxplainer/graphs/contributors).

Please click [here](to-add-after-review) for more information about how to make a contribution to this project.

## Documentation  
Documentation page is ready, will update and publish after double-blind review.

## License
[MIT License](https://github.com/foxplainer/foxplainer/blob/main/LICENSE)

