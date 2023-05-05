# Export pre-trained ML models as source code

The idea behind this library is to export pre-trained ML models to various languages and then run them on embedded platforms.

As now supported are 8 languages and 2 functions.

## Languages
C, dart, go, js, php, python, ruby, rust

## Functions

Uses **LinearRegression()** as model:
```
def export_linear_model(model, lang, ident="\t", fn_name="linear_model", fn_type=None,
	feature_types=[])
```

Uses **DecisionTreeClassifier()** or **DecisionTreeRegressor()** as model:
```
def export_tree(model, lang, ident="\t", fn_name="tree", fn_type=None, feature_types=[])
```

### common params:
- **model** - pre-trained model
- **lang** - export language
- **ident** - string or number of spaces
- **fn_name** - exported function name
- **fn_type** - return type of a function
- **feature_types** - list with types of arguments (features)

## Examples

### DecisionTreeClassifier

```
import pandas as pd
from sklearn.tree import *
from ml_export.sklearn import *

# Let's use xor table as input data
df = pd.DataFrame([
	[0, 0, 0],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 0],
], columns=["a", "b", "c"])

X = df.drop("c", axis=1)
y = df["c"]

# train DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# print trained model in python 
print(export_tree(clf, "python"))

# print trained model in C using all optional parameters
print(export_tree(clf, "C", ident=4, fn_name="xor", fn_type="int", feature_types=["int", "int"]))
```
outputs:
```
def tree(a: float, b: float) -> float:
	if b <= 0.5:
		if a <= 0.5:
			return 0
		else:
			return 1
	else:
		if a <= 0.5:
			return 1
		else:
			return 0

int xor(int a, int b) {
    if(b <= 0.5) {
        if(a <= 0.5) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if(a <= 0.5) {
            return 1;
        } else {
            return 0;
        }
    }
}

```

### LinearRegression

```
import pandas as pd
from sklearn.linear_model import *
from ml_export.sklearn import *

# let's use formula: c = 1 * a + 2 * b + 3
df = pd.DataFrame([
	[0, 0, 3],
	[0, 1, 5],
	[1, 0, 4],
	[1, 1, 6],
], columns=["a", "b", "c"])

X = df.drop("c", axis=1)
y = df["c"]

# train LinearRegression
regr = LinearRegression()
regr.fit(X, y)

# print trained model in go
print(export_linear_model(regr, "go"))
```
outputs:
```
func linear_model(a float32, b float32) float32 {
	return 1.0 * float32(a) + 1.9999999999999998 * float32(b) + 3.0
}
```