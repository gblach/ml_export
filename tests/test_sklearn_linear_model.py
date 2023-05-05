import glob, os, subprocess
import pytest
import pandas as pd
from sklearn.linear_model import *
from ml_export.sklearn import *

df = pd.DataFrame([
	[0, 0, 3],
	[0, 1, 5],
	[1, 0, 4],
	[1, 1, 6],
], columns=["a", "b", "c"])

X = df.drop("c", axis=1)
y = df["c"]

regr = LinearRegression()
regr.fit(X, y)

@pytest.fixture()
def cleanup():
	yield
	for file in glob.glob("code*"):
		os.unlink(file)

code_c = """%s
int main() {
	if(linear_model(0, 0) != 3) return 1;
	if(linear_model(0, 1) != 5) return 2;
	if(linear_model(1, 0) != 4) return 3;
	if(linear_model(1, 1) != 6) return 4;
	return 0;
}
"""

def test_c(cleanup):
	with open("code.c", "w") as f:
		export = export_linear_model(regr, "C")
		f.write(code_c % export)
	rc = subprocess.run(["cc", "-ocode", "code.c"])
	assert rc.returncode == 0
	rc = subprocess.run(["./code"])
	assert rc.returncode == 0

code_dart = """import 'dart:io';
%s
void main() {
	if(linear_model(0, 0).round() != 3) exit(1);
	if(linear_model(0, 1).round() != 5) exit(2);
	if(linear_model(1, 0).round() != 4) exit(3);
	if(linear_model(1, 1).round() != 6) exit(4);
}
"""

def test_dart():
	with open("code.dart", "w") as f:
		export = export_linear_model(regr, "dart")
		f.write(code_dart % export)
	rc = subprocess.run(["dart", "code.dart"])
	assert rc.returncode == 0

code_go = """package main
import "os"
%s
func main() {
	if linear_model(0, 0) != 3 {
		os.Exit(1)
	}
	if linear_model(0, 1) != 5 {
		os.Exit(2)
	}
	if linear_model(1, 0) != 4 {
		os.Exit(3)
	}
	if linear_model(1, 1) != 6 {
		os.Exit(4)
	}
	os.Exit(0)
}
"""

def test_go(cleanup):
	with open("code.go", "w") as f:
		export = export_linear_model(regr, "go")
		f.write(code_go % export)
	rc = subprocess.run(["go", "run", "code.go"])
	assert rc.returncode == 0

code_go_typed = """package main
import "os"
%s
func main() {
	if typed(0, 0) != 3 {
		os.Exit(1)
	}
	if typed(0, 1) != 5 {
		os.Exit(2)
	}
	if typed(1, 0) != 4 {
		os.Exit(3)
	}
	if typed(1, 1) != 6 {
		os.Exit(4)
	}
	os.Exit(0)
}
"""

def test_go_typed():
	with open("code.go", "w") as f:
		export = export_linear_model(regr, "go",
			ident=4, fn_name="typed", fn_type="float64",
			feature_types=["int32", "int32"])
		f.write(code_go_typed % export)
	rc = subprocess.run(["go", "run", "code.go"])
	assert rc.returncode == 0

code_js = """%s
if(linear_model(0, 0) != 3) process.exit(1)
if(linear_model(0, 1) != 5) process.exit(2)
if(linear_model(1, 0) != 4) process.exit(3)
if(linear_model(1, 1) != 6) process.exit(4)
"""

def test_js(cleanup):
	with open("code.js", "w") as f:
		export = export_linear_model(regr, "js")
		f.write(code_js % export)
	rc = subprocess.run(["node", "code.js"])
	assert rc.returncode == 0

code_php = """<?php
%s
if(linear_model(0, 0) != 3) exit(1);
if(linear_model(0, 1) != 5) exit(2);
if(linear_model(1, 0) != 4) exit(3);
if(linear_model(1, 1) != 6) exit(4);
"""

def test_php():
	with open("code.php", "w") as f:
		export = export_linear_model(regr, "php")
		f.write(code_php % export)
	rc = subprocess.run(["php", "code.php"])
	assert rc.returncode == 0

code_python = """%s
import sys
if linear_model(0, 0) != 3: sys.exit(1)
if linear_model(0, 1) != 5: sys.exit(2)
if linear_model(1, 0) != 4: sys.exit(3)
if linear_model(1, 1) != 6: sys.exit(4)
"""

def test_python(cleanup):
	with open("code.py", "w") as f:
		export = export_linear_model(regr, "python")
		f.write(code_python % export)
	rc = subprocess.run(["python", "code.py"])
	assert rc.returncode == 0

code_ruby = """%s
if linear_model(0, 0) != 3
	exit(1)
end
if linear_model(0, 1) != 5
	exit(2)
end
if linear_model(1, 0) != 4
	exit(3)
end
if linear_model(1, 1) != 6
	exit(4)
end
"""

def test_ruby(cleanup):
	with open("code.rb", "w") as f:
		export = export_linear_model(regr, "ruby")
		f.write(code_ruby % export)
	rc = subprocess.run(["ruby", "code.rb"])
	assert rc.returncode == 0

code_rust = """use std::process::ExitCode;
%s
fn main() -> ExitCode {
	if linear_model(0.0, 0.0) != 3.0 {
		return ExitCode::from(1);
	}
	if linear_model(0.0, 1.0) != 5.0 {
		return ExitCode::from(2);
	}
	if linear_model(1.0, 0.0) != 4.0 {
		return ExitCode::from(3);
	}
	if linear_model(1.0, 1.0) != 6.0 {
		return ExitCode::from(4);
	}
	ExitCode::from(0)
}
"""

def test_rust(cleanup):
	with open("code.rs", "w") as f:
		export = export_linear_model(regr, "rust")
		f.write(code_rust % export)
	rc = subprocess.run(["rustc", "-ocode", "code.rs"])
	assert rc.returncode == 0
	rc = subprocess.run(["./code"])
	assert rc.returncode == 0

code_rust_typed = """use std::process::ExitCode;
%s
fn main() -> ExitCode {
	if typed(0, 0) != 3.0 {
		return ExitCode::from(1);
	}
	if typed(0, 1) != 5.0 {
		return ExitCode::from(2);
	}
	if typed(1, 0) != 4.0 {
		return ExitCode::from(3);
	}
	if typed(1, 1) != 6.0 {
		return ExitCode::from(4);
	}
	ExitCode::from(0)
}
"""

def test_rust_typed(cleanup):
	with open("code.rs", "w") as f:
		export = export_linear_model(regr, "rust",
			ident=4, fn_name="typed", fn_type="f64",
			feature_types=["i32", "i32"])
		f.write(code_rust_typed % export)
	rc = subprocess.run(["rustc", "-ocode", "code.rs"])
	assert rc.returncode == 0
	rc = subprocess.run(["./code"])
	assert rc.returncode == 0
