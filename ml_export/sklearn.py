from .lang import LANG

def _fn_begin(model, lang, fn_name, fn_type, feature_types):
	code = ''
	args = []
	for i in range(model.n_features_in_):
		_type = feature_types[i] if i < len(feature_types) else LANG[lang]["default_type"]
		args.append(LANG[lang]["fn_arg"].format(
			name=model.feature_names_in_[i],
			type=_type))
	code += LANG[lang]["fn_begin"].format(
		name=fn_name,
		type=fn_type or LANG[lang]["default_type"],
		args=LANG[lang]["fn_arg_sep"].join(args)) + "\n"
	return code

def _fn_end(lang):
	if None != LANG[lang]["fn_end"]:
		return LANG[lang]["fn_end"] + "\n"
	return ""

def export_linear_model(model, lang, ident="\t", fn_name="linear_model", fn_type=None,
	feature_types=[]):

	if type(ident) is int:
		ident = ' ' * ident
	code = _fn_begin(model, lang, fn_name, fn_type, feature_types)

	pred = ''
	for i in range(model.n_features_in_):
		_type = fn_type or LANG[lang]["default_type"]
		var = LANG[lang]["var"].format(var=model.feature_names_in_[i], type=_type)
		pred += '%s * %s + ' % (model.coef_[i], var)
	pred += str(model.intercept_)
	code += ident + LANG[lang]["return"].format(pred) + "\n"

	code += _fn_end(lang)
	return code

def export_tree(model, lang, ident="\t", fn_name="tree", fn_type=None, feature_types=[],
	*, _i=0, _t=1):

	code = ''

	if 0 == _i:
		if type(ident) is int:
			ident = ' ' * ident
		code += _fn_begin(model, lang, fn_name, fn_type, feature_types)

	if 0 <= model.tree_.feature[_i]:
		feature_name = model.feature_names_in_[model.tree_.feature[_i]]
		threshold = model.tree_.threshold[_i]
		code += (ident * _t) + LANG[lang]["if"].format(feature_name, threshold) + "\n"
		code += export_tree(model, lang, ident=ident,
			_i=model.tree_.children_left[_i], _t=_t+1)
		code += (ident * _t) + LANG[lang]["else"] + "\n"
		code += export_tree(model, lang, ident=ident,
			_i=model.tree_.children_right[_i], _t=_t+1)
		if LANG[lang]["endif"]:
			code += (ident * _t) + LANG[lang]["endif"] + "\n"
	else:
		if hasattr(model, "classes_"):
			pred = model.classes_[model.tree_.value[_i].argmax()]
		else:
			pred = model.tree_.value[_i].max()
		code += (ident * _t) + LANG[lang]["return"].format(pred) + "\n"

	if 0 == _i:
		code += _fn_end(lang)
	return code
