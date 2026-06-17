import warnings
from sklearn.metrics import confusion_matrix, make_scorer


DEFAULT_ZERO_DIVISION = "warn"


def _zero_division_handler(metric_name, zero_division):
	if zero_division == "warn":
		warnings.warn(
			f"{metric_name} is ill-defined due to division by zero. "
			"Setting result to 0.0.",
			UserWarning,
			stacklevel=3
		)
		return 0.0

	return float(zero_division)


def specificity(y_true, y_pred, sample_weight=None, zero_division=DEFAULT_ZERO_DIVISION):
	tn, fp, fn, tp = confusion_matrix(
		y_true,
		y_pred,
		sample_weight=sample_weight
	).ravel()

	denominator = tn + fp

	if denominator == 0:
		return _zero_division_handler("Specificity", zero_division)

	return tn / denominator


def negative_predictive_value(
	y_true,
	y_pred,
	sample_weight=None,
	zero_division=DEFAULT_ZERO_DIVISION
):
	tn, fp, fn, tp = confusion_matrix(
		y_true,
		y_pred,
		sample_weight=sample_weight
	).ravel()

	denominator = tn + fn

	if denominator == 0:
		return _zero_division_handler(
			"Negative Predictive Value",
			zero_division
		)

	return tn / denominator


specificity_scorer = make_scorer(
	specificity,
	greater_is_better=True,
	zero_division=DEFAULT_ZERO_DIVISION
)


npv_scorer = make_scorer(
	negative_predictive_value,
	greater_is_better=True,
	zero_division=DEFAULT_ZERO_DIVISION
)


SCORERS = {
	"specificity": specificity_scorer,
	"true_negative_rate": specificity_scorer,

	"npv": npv_scorer,
	"negative_predictive_value": npv_scorer,
}

