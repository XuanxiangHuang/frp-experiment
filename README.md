# Experiments of Feature Necessity Problem 
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
---
# Experiment for SDD

## Benchmarks:
[Density Estimation Datasets](https://github.com/UCLA-StarAI/Density-Estimation-Datasets)

## Pre-trained Models

All SDD models are in `sdds` folder. 
They were learned by using [LearnSDD](https://github.com/ML-KULeuven/LearnSDD) package, 
with parameter `maxEdges=20000`.

## Algorithm

Please check `sdd_feature_relevancy.py`.

## Examples:
Generating tested instances/features:
```
python3 gen_samples.py -bench density-estimation.txt -inst 100
python3 gen_samples.py -bench density-estimation.txt -feat 100
```

## Reproduce
Run this command for reproducing the experiments:
```
python3 reproduce.py -bench density-estimation.txt
```

## Results
All the reported results in `results` folder:
1. `sdd_frp_log.txt` output of `reproduce.py` script.
2. `sdd_frp.txt`: the reported results.

# Experiment for DLN

## Training

5 scripts for training DLN models in `DLN_training` folder.
Run this command to train DLN for dataset australian:

```
cd DLN_training

python3 trainDLN_australian.py
```
Besides, for evaluating runtime performance of DLN's predict()
function, run:
```
python3 clf_time_test.py
```

## Pre-trained Models

5 pretrained models in the `DLNs` folder.
DLN model summary is in `DLN_training/DLN_model_info.txt`
and `DLN_training/DLN_clf_time.txt`:
1. accuracy.
2. number of parameters.
3. max and median CPU time calling predict() function.
 
## Algorithm

Please check `mono_feature_relevancy.py`.

## Tested instances/features

You can find tested instances/features for deciding feature relevancy problem
in `samples` folder.
subfolder `clf_time_test` contains 10000 randomly picked data points
for evaluating runtime performance of DLN model (note that there is no label column).
Besides, you can generate tested features by running:

```
python3 gen_test_feats.py
```
You can also pick tested datapoints from feature space by running:
```
python3 gen_predict_test_insts.py
```

## Reproduce

Run this command for reproducing the experiments:
```
python3 reproduce.py
```

## Results
All the reported results in `results` folder:
1. subfolder `calls`: number of calls to the SAT solver/DLN's predict() for each tested instance. 
2. subfolder `runtime`: accumulated CPU time calling SAT solver/DLN's predict() for each tested instance. And the runtime for each tested instance.
3. `mono_frp_log.txt` output of `reproduce.py` script.
4. `mono_frp.txt`: the reported results.

## Citation
```
@inproceedings{huang2023feature,
  title={Feature necessity \& relevancy in ML classifier explanations},
  author={Huang, Xuanxiang and Cooper, Martin C and Morgado, Antonio and Planes, Jordi and Marques-Silva, Joao},
  booktitle={Tools and Algorithms for the Construction and Analysis of Systems: 29th International Conference, TACAS 2023, Held as Part of the European Joint Conferences on Theory and Practice of Software, ETAPS 2022, Paris, France, April 22--27, 2023, Proceedings, Part I},
  pages={167--186},
  year={2023},
  organization={Springer}
}
```
