def is_float(x):
	return isinstance(x, (float, np.floating))


def is_dict(x):
	return isinstance(x, dict)


def is_list(x):
	return isinstance(x, list)


def is_set(x):
	return isinstance(x, set)


def is_string(x):
	return isinstance(x, str)


def is_sequence(x):
	return isinstance(x, (list, tuple, set))

