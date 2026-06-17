from .registry import (
	make_builder,
	make_safe_builder,
	_register_estimator,
	register_estimator,
	resolve_estimator,
	build_estimator,
	estimator_kind,
	available_kinds,
	available_estimators,
	estimator_info
)


__all__ = [
	"make_builder",
	"make_safe_builder",
	"_register_estimator",
	"register_estimator",
	"resolve_estimator",
	"build_estimator",
	"estimator_kind",
	"available_kinds",
	"available_estimators",
	"estimator_info"
]


_loaded = False

def _ensure_loaded():
	global _loaded
	if _loaded:
		return
	
	from ._sklearn import load_sklearn_estimators
	load_sklearn_estimators()

	from ._third_party import load_third_party_estimators
	load_third_party_estimators()
	
	_loaded = True


_ensure_loaded()

