import inspect

from ...utils import (
	assert_string
)

_REGISTRY = {}


def make_builder(cls, validate_kwargs=False, **defaults):
	def builder(**params):
		final = {**defaults, **params}

		if validate_kwargs:
			sig = inspect.signature(cls.__init__)
			allowed = set(sig.parameters.keys()) - {"self"}

			final = {
				key: value
				for key, value in final.items()
				if key in allowed
			}

		return cls(**final)

	return builder


def make_safe_builder(cls, **defaults):
	return make_builder(cls, validate_kwargs=True, **defaults)


def _register_estimator(
	name,
	builder=None,
	cls=None,
	kind="other",
	auto_build=True,
):
	assert_string(name, "name must be a string")

	if builder is not None:
		pass
	elif cls is not None:
		if not auto_build:
			raise ValueError(
				"builder is required when auto_build=False"
			)

		builder = make_builder(cls)
	else:
		raise ValueError(
			"Either 'builder' or 'cls' must be provided"
		)

	bucket = _REGISTRY.setdefault(kind, {})

	if name in bucket:
		raise ValueError(
			f"'{name}' already registered in '{kind}'"
		)

	bucket[name] = {
		"builder": builder,
		"cls": cls,
	}

	return bucket[name]


def register_estimator(name, builder=None, cls=None):
	return _register_estimator(name, builder, cls, kind="user")


def resolve_estimator(name, kind=None):
	if kind:
		if kind not in _REGISTRY or name not in _REGISTRY[kind]:
			raise KeyError(f"Unknown estimator: {name} in {kind}")
		return _REGISTRY[kind][name]["builder"]

	for bucket in _REGISTRY.values():
		if name in bucket:
			return bucket[name]["builder"]

	raise KeyError(f"Unknown estimator: {name}")


def build_estimator(name, kind=None, **params):
	builder = resolve_estimator(name, kind=kind)
	return builder(**params)


def estimator_kind(name):
	matches = []

	for kind, bucket in _REGISTRY.items():
		if name in bucket:
			matches.append(kind)

	if not matches:
		raise KeyError(f"Unknown estimator: {name}")

	return tuple(sorted(matches))


def available_kinds():
	return tuple(sorted(_REGISTRY.keys()))


def available_estimators(kind=None):
	if kind is not None:
		if kind not in _REGISTRY:
			raise KeyError(f"Unknown estimator kind: {kind}")
		return tuple(sorted(_REGISTRY[kind].keys()))

	all_names = set()
	for bucket in _REGISTRY.values():
		all_names.update(bucket.keys())

	return tuple(sorted(all_names))


def estimator_info(name):
	matches = []

	for kind, bucket in _REGISTRY.items():
		if name in bucket:
			info = bucket[name]
			matches.append({
				"name": name,
				"kind": kind,
				"class": info["cls"],
			})

	if not matches:
		raise KeyError(f"Unknown estimator: {name}")

	return matches

