LANG = {
	"C": {
		"fn_begin": "{type} {name}({args}) {{",
		"fn_arg": "{type} {name}",
		"fn_arg_sep": ", ",
		"fn_end": "}",
		"if": "if({} <= {}) {{",
		"else": "} else {",
		"endif": "}",
		"return": "return {};",
		"var": '{var}',
		"default_type": "float",
	},
	"dart": {
		"fn_begin": "{type} {name}({args}) {{",
		"fn_arg": "{type} {name}",
		"fn_arg_sep": ", ",
		"fn_end": "}",
		"if": "if({} <= {}) {{",
		"else": "} else {",
		"endif": "}",
		"return": "return {};",
		"var": '{var}',
		"default_type": "double",
	},
	"go": {
		"fn_begin": "func {name}({args}) {type} {{",
		"fn_arg": "{name} {type}",
		"fn_arg_sep": ", ",
		"fn_end": "}",
		"if": "if {} <= {} {{",
		"else": "} else {",
		"endif": "}",
		"return": "return {}",
		"var": '{type}({var})',
		"default_type": "float32",	
	},
	"js": {
		"fn_begin": "function {name}({args}) {{",
		"fn_arg": "{name}",
		"fn_arg_sep": ", ",
		"fn_end": "}",
		"if": "if({} <= {}) {{",
		"else": "} else {",
		"endif": "}",
		"return": "return {};",
		"var": '{var}',
		"default_type": None,	
	},
	"php": {
		"fn_begin": "function {name}({args}) {{",
		"fn_arg": "${name}",
		"fn_arg_sep": ", ",
		"fn_end": "}",
		"if": "if(${} <= {}) {{",
		"else": "} else {",
		"endif": "}",
		"return": "return {};",
		"var": '${var}',
		"default_type": None,
	},
	"python": {
		"fn_begin": "def {name}({args}) -> {type}:",
		"fn_arg": "{name}: {type}",
		"fn_arg_sep": ", ",
		"fn_end": None,
		"if": "if {} <= {}:",
		"else": "else:",
		"endif": None,
		"return": "return {}",
		"var": '{var}',
		"default_type": "float",
	},
	"ruby": {
		"fn_begin": "def {name}({args})",
		"fn_arg": "{name}",
		"fn_arg_sep": ", ",
		"fn_end": "end",
		"if": "if {} <= {}",
		"else": "else",
		"endif": "end",
		"return": "{}",
		"var": '{var}',
		"default_type": None,	
	},
	"rust": {
		"fn_begin": "fn {name}({args}) -> {type} {{",
		"fn_arg": "{name}: {type}",
		"fn_arg_sep": ", ",
		"fn_end": "}",
		"if": "if {} <= {} {{",
		"else": "} else {",
		"endif": "}",
		"return": "{}",
		"var": '{var} as {type}',
		"default_type": "f32",
	},
}
