import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class ValueReplacer(BaseEstimator, TransformerMixin):
	def __init__(self, values_to_replace, replacement):
		self.values_to_replace = values_to_replace
		self.replacement = replacement

	def fit(self, X, y=None):
		X = validate_data(self, X, reset=True, ensure_all_finite="allow-nan")

		self.replace_values_ = np.asarray(self.values_to_replace)
		self.replace_set_ = set(self.replace_values_)

		return self

	def transform(self, X):
		check_is_fitted(self, "replace_set_")

		X = validate_data(self, X, reset=True, ensure_all_finite="allow-nan")

		X = np.asarray(X).copy()

		mask = np.isin(X, self.replace_values_)
		X[mask] = self.replacement

		return X

	def get_feature_names_out(self, input_features=None):
		check_is_fitted(self, "replace_set_")

		if hasattr(self, "feature_names_in_"):
			return np.asarray(self.feature_names_in_)

		if input_features is not None:
			return np.asarray(input_features)

		raise ValueError("Unable to determine feature names.")

