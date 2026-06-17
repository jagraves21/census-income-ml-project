def ensure_dict(x, default=None):
	if x is None:
		return {} if default is None else dict(default)
	return dict(x)


def ensure_list(x, default=None):
	if x is None:
		return [] if default is None else list(default)
	return list(x)


def ensure_set(x, default=None):
	if x is None:
		return set() if default is None else set(default)
	return set(x)


def ensure_tuple(x, default=None):
	if x is None:
		return () if default is None else tuple(default)
	return tuple(x)

