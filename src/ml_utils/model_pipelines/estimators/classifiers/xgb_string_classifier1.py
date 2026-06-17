import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from xgboost import XGBClassifier


class XGBStringClassifier(BaseEstimator, ClassifierMixin):
	def __init__(
		self,
		n_estimators=100,
		max_depth=6,
		learning_rate=0.3,
		subsample=1.0,
		colsample_bytree=1.0,
		reg_alpha=0.0,
		reg_lambda=1.0,
		random_state=None,
		n_jobs=None,
		**xgb_kwargs
	):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.learning_rate = learning_rate
		self.subsample = subsample
		self.colsample_bytree = colsample_bytree
		self.reg_alpha = reg_alpha
		self.reg_lambda = reg_lambda
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.xgb_kwargs = xgb_kwargs

	def _build_model(self):
		return XGBClassifier(
			n_estimators=self.n_estimators,
			max_depth=self.max_depth,
			learning_rate=self.learning_rate,
			subsample=self.subsample,
			colsample_bytree=self.colsample_bytree,
			reg_alpha=self.reg_alpha,
			reg_lambda=self.reg_lambda,
			random_state=self.random_state,
			n_jobs=self.n_jobs,
			**self.xgb_kwargs
		)

	def fit(self, X, y, sample_weight=None):
		self.encoder_ = LabelEncoder()
		y_encoded = self.encoder_.fit_transform(y)

		self.model_ = self._build_model()
		self.model_.fit(
			X,
			y_encoded,
			sample_weight=sample_weight
		)

		self.classes_ = self.encoder_.classes_
		self.n_classes_ = len(self.classes_)

		return self

	def predict(self, X):
		check_is_fitted(self, "model_")

		preds = self.model_.predict(X)
		preds = np.asarray(preds, dtype=int)

		return self.encoder_.inverse_transform(preds)

	def predict_proba(self, X):
		check_is_fitted(self, "model_")
		return self.model_.predict_proba(X)

	def score(self, X, y):
		from sklearn.metrics import accuracy_score

		return accuracy_score(y, self.predict(X))

	def get_params(self, deep=True):
		return {
			"n_estimators": self.n_estimators,
			"max_depth": self.max_depth,
			"learning_rate": self.learning_rate,
			"subsample": self.subsample,
			"colsample_bytree": self.colsample_bytree,
			"reg_alpha": self.reg_alpha,
			"reg_lambda": self.reg_lambda,
			"random_state": self.random_state,
			"n_jobs": self.n_jobs,
			**self.xgb_kwargs
		}

	def set_params(self, **params):
		for key, value in params.items():
			if key in self.__dict__:
				setattr(self, key, value)
			else:
				self.xgb_kwargs[key] = value

		return self

