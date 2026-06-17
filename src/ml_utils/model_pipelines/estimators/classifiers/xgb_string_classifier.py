import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBStringClassifier(ClassifierMixin, BaseEstimator):
	def __init__(self, **xgb_params):
		self.xgb_params = xgb_params

	def fit(self, X, y, sample_weight=None, **fit_params):
		self.encoder_ = LabelEncoder()

		y_encoded = self.encoder_.fit_transform(y)

		self.model_ = XGBClassifier(**self.xgb_params)

		self.model_.fit(
			X,
			y_encoded,
			sample_weight=sample_weight,
			**fit_params
		)

		self.classes_ = self.encoder_.classes_

		return self

	def predict(self, X):
		preds = self.model_.predict(X)
		preds = np.asarray(preds, dtype=int)

		return self.encoder_.inverse_transform(preds)

	def predict_proba(self, X):
		return self.model_.predict_proba(X)

	def get_params(self, deep=True):
		return self.xgb_params.copy()

	def set_params(self, **params):
		self.xgb_params = {**self.xgb_params, **params}
		return self

