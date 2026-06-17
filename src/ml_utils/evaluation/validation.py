import logging
import inspect
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import check_cv
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline

from .metrics import SCORERS


logger = logging.getLogger(__name__)


def _get_estimator_name(estimator):
	if isinstance(estimator, Pipeline):
		step_name, final_estimator = estimator.steps[-1]
		return f"Pipeline({step_name}={type(final_estimator).__name__})"

	return type(estimator).__name__


def _supports_sample_weight(estimator):
	if isinstance(estimator, Pipeline):
		_, estimator = estimator.steps[-1]

	fit_sig = inspect.signature(estimator.fit)
	return "sample_weight" in fit_sig.parameters


def _get_fit_kwargs(estimator, train_sample_weight):
	if train_sample_weight is None:
		return {}

	if isinstance(estimator, Pipeline):
		step_name, final_estimator = estimator.steps[-1]

		fit_sig = inspect.signature(final_estimator.fit)

		if "sample_weight" in fit_sig.parameters:
			return {
				f"{step_name}__sample_weight": train_sample_weight
			}

		return {}

	fit_sig = inspect.signature(estimator.fit)

	if "sample_weight" in fit_sig.parameters:
		return {"sample_weight": train_sample_weight}

	return {}


def _resolve_scorer(s):
	if callable(s):
		return s

	if isinstance(s, str):
		if s in SCORERS:
			return SCORERS[s]

		return get_scorer(s)

	raise ValueError(f"Invalid scorer: {s}")


def _build_scorers(scoring):
	if scoring is None:
		return {"score": _resolve_scorer("accuracy")}

	if isinstance(scoring, str):
		return {scoring: _resolve_scorer(scoring)}

	if callable(scoring):
		return {"score": scoring}

	return {
		name: _resolve_scorer(s)
		for name, s in scoring.items()
	}


def _safe_score(scorer, estimator, X, y, eval_sample_weight, name, warned):
	try:
		return scorer(
			estimator,
			X,
			y,
			sample_weight=eval_sample_weight,
		)

	except TypeError:
		if eval_sample_weight is not None and name not in warned:
			logger.warning(
				f"Scorer '{name}' does not support sample_weight. "
				f"Falling back to unweighted scoring."
			)

			warned.add(name)

		return scorer(estimator, X, y)


def fit_model(
	estimator,
	X,
	y,
	sample_weight=None,
):
	if sample_weight is not None:
		sample_weight = np.asarray(sample_weight)

	model = clone(estimator)

	fit_kwargs = _get_fit_kwargs(
		model,
		sample_weight,
	)

	model_name = _get_estimator_name(model)

	if (
		sample_weight is not None
		and not fit_kwargs
	):
		logger.warning(
			f"{model_name} does not support "
			f"sample_weight in fit(). Weights will be ignored."
		)

	model.fit(X, y, **fit_kwargs)

	return model


def weighted_cross_validate(
	estimator,
	X,
	y,
	*,
	cv=5,
	scoring=None,
	sample_weight=None,
	train_sample_weight=None,
	eval_sample_weight=None,
	return_train_score=False,
	return_estimator=False,
):
	if train_sample_weight is None:
		train_sample_weight = sample_weight

	if eval_sample_weight is None:
		eval_sample_weight = sample_weight

	cv = check_cv(
		cv,
		y,
		classifier=getattr(estimator, "_estimator_type", None) == "classifier"
	)

	scorers = _build_scorers(scoring)

	X_df = hasattr(X, "iloc")
	y_df = hasattr(y, "iloc")

	if train_sample_weight is not None:
		train_sample_weight = np.asarray(train_sample_weight)

	if eval_sample_weight is not None:
		eval_sample_weight = np.asarray(eval_sample_weight)

	results = {f"test_{k}": [] for k in scorers}

	if return_train_score:
		results.update({f"train_{k}": [] for k in scorers})

	if return_estimator:
		results["estimator"] = []

	warned_scorers = set()

	for train_idx, test_idx in cv.split(X, y):
		if X_df:
			X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
		else:
			X_train, X_test = X[train_idx], X[test_idx]

		if y_df:
			y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
		else:
			y_train, y_test = y[train_idx], y[test_idx]

		sw_train = None
		sw_test = None

		if train_sample_weight is not None:
			sw_train = train_sample_weight[train_idx]

		if eval_sample_weight is not None:
			sw_test = eval_sample_weight[test_idx]

		import time
		from datetime import timedelta

		start = time.perf_counter()

		model = fit_model(
			estimator,
			X_train,
			y_train,
			sample_weight=sw_train,
		)

		end = time.perf_counter()
		elapsed = end - start

		print(f"  {timedelta(seconds=elapsed)}")

		if return_estimator:
			results["estimator"].append(model)

		for name, scorer in scorers.items():
			results[f"test_{name}"].append(
				_safe_score(
					scorer,
					model,
					X_test,
					y_test,
					sw_test,
					name,
					warned_scorers,
				)
			)

			if return_train_score:
				results[f"train_{name}"].append(
					_safe_score(
						scorer,
						model,
						X_train,
						y_train,
						sw_train,
						name,
						warned_scorers,
					)
				)

	for result_name in results:
		if (
			result_name.startswith("test_")
			or result_name.startswith("train_")
		):
			results[result_name] = np.asarray(results[result_name])

	return results


def get_model_performance(
	df,
	feature_columns,
	target_column,
	models,
	cv=5,
	metrics=None,
	return_folds=False,
	sample_weight=None,
	train_sample_weight=None,
	eval_sample_weight=None,
):
	if train_sample_weight is None:
		train_sample_weight = sample_weight

	if eval_sample_weight is None:
		eval_sample_weight = sample_weight

	X = df[feature_columns]
	y = df[target_column]

	results = []
	for name, model in models.items():
		print(name)

		cv_output = weighted_cross_validate(
			model,
			X=X,
			y=y,
			cv=cv,
			scoring=metrics,
			train_sample_weight=train_sample_weight,
			eval_sample_weight=eval_sample_weight,
		)

		row = {}

		for key, values in cv_output.items():
			if not key.startswith("test_"):
				continue

			metric = key.replace("test_", "")
			values = np.asarray(values)

			row = [
				name, metric,
				values.mean(), values.std(), values.min(), values.max(),
			]

			if return_folds:
				row.extend(values)

			results.append(row)

	columns = ["model", "metric", "mean", "std", "min", "max"]
	if return_folds:
		for ii in range(len(results[0]) - len(columns)):
			columns.append(f"fold_{ii+1}")
	
	df_results = pd.DataFrame(results, columns=columns)

	return df_results

