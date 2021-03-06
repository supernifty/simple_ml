# Simple machine learning

A simple wrapper around scikit-learn for quick prototyping

## Installation

* Tested with Python 3

```
module load Python/3.6.4-intel-2017.u2
python -m venv ml-venv
pip install -r requirements.txt
```

## Usage
```
python ml.py --verbose --X X.tsv --y y.tsv
```

### Relationships

Linear regression + correlation
```
python simple_ml/explore.py --xs "Sepal Length" "Petal Length" --y "Sepal Width" --delimiter ',' < test/iris.data
```

Correlation matrix
```
python simple_ml/correlation.py --cols "Sepal Length" "Sepal Width" "Petal Length" "Petal Width" --delimiter ',' < test/iris.data
```
