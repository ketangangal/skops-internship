# skops-internship Task

1. Create a python environment[1] and install `scikit-learn` version `1.0` in that environment.
2. Using that environment, create a `LogisticRegression` model[2] and fit it on the Iris dataset[3].
3. Save the trained model using `pickle`[4] or `joblib`[5].
4. Create a second environment, and install `scikit-learn` version `1.1` in it.
5. Try loading the model you saved in step 3 in this second environment.

## Steps to replicate 
### Environment - 1
Create Conda env with version python 3.9
```bash
conda create -p ./env-1 python=3.9 -y
conda activate ./env-1
```
```bash
pip install -r requirements-env-1.txt
```
```python
python logistic-main.py
```
### Environment - 2
Create Conda env with version python 3.9
```bash
conda create -p ./env-2 python=3.9 -y
conda activate ./env-2
```
```bash
pip install -r requirements-env-2.txt
```
```python
python logistic-infer.py
```
## Question:

Is there a warning or error you receive while trying to load the model? If yes, what exactly is it.
## Warnings Received 

UserWarning: Trying to unpickle estimator LogisticRegression from version 1.0 when using version 1.1.0.
This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:

https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations

## Reason For Warnings
1. Assume that in scikit learn 1.1 some methods or functionality are present which were not present in scikit learn 1.0. Loading a saved model for another version will cause the error since the functionality is not present in the older/higher version. 

2. Dependency Issues

3. While models saved using one version of scikit-learn might load in other versions, this is entirely unsupported and inadvisable.


## Solutions 
You might like to manually output the parameters of your learned model so that you can use them directly in future version of scikit-learn.

