import glob, os, subprocess
import pytest
import pandas as pd
from sklearn.tree import *
from ml_export.sklearn import *

df = pd.DataFrame([
	[0, 0, 0],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 0],
], columns=["a", "b", "c"])

X = df.drop("c", axis=1)
y = df["c"]

clf = DecisionTreeClassifier()
clf.fit(X, y)

regr = DecisionTreeRegressor()
regr.fit(X, y / 2)

@pytest.fixture()
def cleanup():
	yield
	for file in glob.glob("code*"):
		os.unlink(file)

code_c = """%s
int main() {
	if(tree(0, 0) != 0) return 1;
	if(tree(0, 1) != 1) return 2;
	if(tree(1, 0) != 1) return 3;
	if(tree(1, 1) != 0) return 4;
	return 0;
}
"""

def test_c(cleanup):
	with open("code.c", "w") as f:
		export = export_tree(clf, "C")
		f.write(code_c % export)
	rc = subprocess.run(["cc", "-ocode", "code.c"])
	assert rc.returncode == 0
	rc = subprocess.run(["./code"])
	assert rc.returncode == 0

code_c_xor = """%s
int main() {
	if(xor(0, 0) != 0) return 1;
	if(xor(0, 1) != 1) return 2;
	if(xor(1, 0) != 1) return 3;
	if(xor(1, 1) != 0) return 4;
	return 0;
}
"""

def test_c_xor(
	):
	with open("code.c", "w") as f:
		export = export_tree(clf, "C",
			ident=4, fn_name='xor', fn_type='int',
			feature_types=['int', 'int'])
		f.write(code_c_xor % export)
	rc = subprocess.run(["cc", "-ocode", "code.c"])
	assert rc.returncode == 0
	rc = subprocess.run(["./code"])
	assert rc.returncode == 0

code_dart = """import 'dart:io';
%s
void main() {
	if(tree(0, 0) != 0) exit(1);
	if(tree(0, 1) != 1) exit(2);
	if(tree(1, 0) != 1) exit(3);
	if(tree(1, 1) != 0) exit(4);
}
"""

def test_dart(cleanup):
	with open("code.dart", "w") as f:
		export = export_tree(clf, "dart")
		f.write(code_dart % export)
	rc = subprocess.run(["dart", "code.dart"])
	assert rc.returncode == 0

code_go = """package main
import "os"
%s
func main() {
	if tree(0, 0) != 0 {
		os.Exit(1)
	}
	if tree(0, 1) != 1 {
		os.Exit(2)
	}
	if tree(1, 0) != 1 {
		os.Exit(3)
	}
	if tree(1, 1) != 0 {
		os.Exit(4)
	}
	os.Exit(0)
}
"""

def test_go(cleanup):
	with open("code.go", "w") as f:
		export = export_tree(clf, "go")
		f.write(code_go % export)
	rc = subprocess.run(["go", "run", "code.go"])
	assert rc.returncode == 0

code_js = """%s
if(tree(0, 0) != 0) process.exit(1)
if(tree(0, 1) != 1) process.exit(2)
if(tree(1, 0) != 1) process.exit(3)
if(tree(1, 1) != 0) process.exit(4)
"""

def test_js(cleanup):
	with open("code.js", "w") as f:
		export = export_tree(clf, "js")
		f.write(code_js % export)
	rc = subprocess.run(["node", "code.js"])
	assert rc.returncode == 0

code_php = """<?php
%s
if(tree(0, 0) != 0) exit(1);
if(tree(0, 1) != 1) exit(2);
if(tree(1, 0) != 1) exit(3);
if(tree(1, 1) != 0) exit(4);
"""

def test_php(cleanup):
	with open("code.php", "w") as f:
		export = export_tree(clf, "php")
		f.write(code_php % export)
	rc = subprocess.run(["php", "code.php"])
	assert rc.returncode == 0

code_python = """%s
import sys
if tree(0, 0) != 0: sys.exit(1)
if tree(0, 1) != 1: sys.exit(2)
if tree(1, 0) != 1: sys.exit(3)
if tree(1, 1) != 0: sys.exit(4)
"""

def test_python(cleanup):
	with open("code.py", "w") as f:
		export = export_tree(clf, "python")
		f.write(code_python % export)
	rc = subprocess.run(["python", "code.py"])
	assert rc.returncode == 0

code_python_regr = """%s
import sys
if tree(0, 0) != 0.0: sys.exit(1)
if tree(0, 1) != 0.5: sys.exit(2)
if tree(1, 0) != 0.5: sys.exit(3)
if tree(1, 1) != 0.0: sys.exit(4)
"""

def test_python_regr(cleanup):
	with open("code.py", "w") as f:
		export = export_tree(regr, "python")
		f.write(code_python_regr % export)
	rc = subprocess.run(["python", "code.py"])
	assert rc.returncode == 0

code_ruby = """%s
if tree(0, 0) != 0
	exit(1)
end
if tree(0, 1) != 1
	exit(2)
end
if tree(1, 0) != 1
	exit(3)
end
if tree(1, 1) != 0
	exit(4)
end
"""

def test_ruby(cleanup):
	with open("code.rb", "w") as f:
		export = export_tree(clf, "ruby")
		f.write(code_ruby % export)
	rc = subprocess.run(["ruby", "code.rb"])
	assert rc.returncode == 0

code_rust = """use std::process::ExitCode;
%s
fn main() -> ExitCode {
	if tree(0.0, 0.0) != 0 {
		return ExitCode::from(1);
	}
	if tree(0.0, 1.0) != 1 {
		return ExitCode::from(2);
	}
	if tree(1.0, 0.0) != 1 {
		return ExitCode::from(3);
	}
	if tree(1.0, 1.0) != 0 {
		return ExitCode::from(4);
	}
	ExitCode::from(0)
}
"""

def test_rust(cleanup):
	with open("code.rs", "w") as f:
		export = export_tree(clf, "rust", fn_type="i32")
		f.write(code_rust % export)
	rc = subprocess.run(["rustc", "-ocode", "code.rs"])
	assert rc.returncode == 0
	rc = subprocess.run(["./code"])
	assert rc.returncode == 0
