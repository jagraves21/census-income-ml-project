from .typing import (
	is_dict,
	is_list,
	is_set,
	is_string
)


def assert_dict(x, msg="expected dict"):
	if not is_dict(x):
		raise TypeError(msg)
	return x


def assert_list(x, msg="expected list"):
	if not is_list(x):
		raise TypeError(msg)
	return x


def assert_set(x, msg="expected set"):
	if not is_set(x):
		raise TypeError(msg)
	return x


def assert_string(x, msg="expected string"):
	if not is_string(x):
		raise TypeError(msg)
	return x

