import numpy as np

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class OneHotEncoderWithIgnore(BaseEstimator, TransformerMixin):

	def __init__(
		self,
		*,
		categories="auto",
		ignore_categories=None,
		drop=None,
		sparse_output=True,
		dtype=np.float64,
		handle_unknown="error",
	):
		self.categories = categories
		self.ignore_categories = ignore_categories
		self.drop = drop
		self.sparse_output = sparse_output
		self.dtype = dtype
		self.handle_unknown = handle_unknown

	def fit(self, X, y=None):
		X = self._validate_X(X)

		self.n_features_in_ = X.shape[1]

		if self.handle_unknown not in {"error", "ignore"}:
			raise ValueError(
				"handle_unknown must be either 'error' or 'ignore'"
			)

		self.categories_ = []
		self.category_mapping_ = []

		if self.categories == "auto":
			for i in range(X.shape[1]):
				column = X[:, i]

				categories = self._extract_categories(column)

				self.categories_.append(categories)

				self.category_mapping_.append(
					{
						self._hashable_key(cat): idx
						for idx, cat in enumerate(categories)
					}
				)
		else:
			if len(self.categories) != X.shape[1]:
				raise ValueError(
					"categories length must match number of features"
				)

			for feature_categories in self.categories:
				categories = np.asarray(
					feature_categories,
					dtype=object,
				)

				self.categories_.append(categories)

				self.category_mapping_.append(
					{
						self._hashable_key(cat): idx
						for idx, cat in enumerate(categories)
					}
				)

		self.drop_idx_ = self._compute_drop_idx()
		self._compute_feature_layout()

		return self

	def transform(self, X):
		check_is_fitted(self, "categories_")

		X = self._validate_X(X)

		if X.shape[1] != self.n_features_in_:
			raise ValueError(
				f"Expected {self.n_features_in_} features, got {X.shape[1]}"
			)

		indices = []
		indptr = [0]
		data = []

		for row_idx in range(X.shape[0]):
			row_nnz = 0

			for feature_idx in range(X.shape[1]):
				value = X[row_idx, feature_idx]

				mapping = self.category_mapping_[feature_idx]

				key = self._hashable_key(value)

				if self._is_ignored(value):
					continue

				if key not in mapping:
					if self.handle_unknown == "error":
						raise ValueError(
							f"Unknown category {value!r} in feature "
							f"{feature_idx}"
						)

					continue

				category_idx = mapping[key]

				drop_idx = self.drop_idx_[feature_idx]

				if drop_idx is not None:
					if category_idx == drop_idx:
						continue

					if category_idx > drop_idx:
						category_idx -= 1

				global_idx = (
					self.feature_offsets_[feature_idx]
					+ category_idx
				)

				indices.append(global_idx)
				data.append(1)
				row_nnz += 1

			indptr.append(indptr[-1] + row_nnz)

		matrix = sparse.csr_matrix(
			(data, indices, indptr),
			shape=(X.shape[0], self.n_output_features_),
			dtype=self.dtype,
		)

		if self.sparse_output:
			return matrix

		return matrix.toarray()

	def fit_transform(self, X, y=None):
		return self.fit(X, y).transform(X)

	def inverse_transform(self, X):
		check_is_fitted(self, "categories_")

		if sparse.issparse(X):
			X = X.toarray()

		X = np.asarray(X)

		result = np.empty(
			(X.shape[0], self.n_features_in_),
			dtype=object,
		)

		for feature_idx in range(self.n_features_in_):
			start = self.feature_offsets_[feature_idx]
			end = self.feature_offsets_[feature_idx + 1]

			block = X[:, start:end]

			categories = self.categories_[feature_idx]
			drop_idx = self.drop_idx_[feature_idx]

			for row_idx in range(X.shape[0]):
				row = block[row_idx]

				if np.all(row == 0):
					if drop_idx is not None:
						result[row_idx, feature_idx] = categories[drop_idx]
					else:
						result[row_idx, feature_idx] = None

					continue

				active = np.argmax(row)

				if drop_idx is not None and active >= drop_idx:
					active += 1

				result[row_idx, feature_idx] = categories[active]

		return result

	def get_feature_names_out(self, input_features=None):
		check_is_fitted(self, "categories_")

		if input_features is None:
			input_features = [
				f"x{i}" for i in range(self.n_features_in_)
			]

		names = []

		for feature_idx, feature_name in enumerate(input_features):
			categories = self.categories_[feature_idx]
			drop_idx = self.drop_idx_[feature_idx]

			for category_idx, category in enumerate(categories):
				if drop_idx is not None and category_idx == drop_idx:
					continue

				names.append(f"{feature_name}_{category}")

		return np.asarray(names, dtype=object)

	def _validate_X(self, X):
		X = check_array(
			X,
			dtype=None,
			ensure_2d=True,
			ensure_all_finite="allow-nan",
		)

		return np.asarray(X, dtype=object)

	def _extract_categories(self, column):
		column = np.asarray(column, dtype=object)

		seen = set()
		categories = []

		for value in column:
			if self._is_ignored(value):
				continue

			key = self._hashable_key(value)

			if key not in seen:
				seen.add(key)
				categories.append(value)

		try:
			return np.asarray(sorted(categories), dtype=object)
		except TypeError:
			return np.asarray(categories, dtype=object)

	def _hashable_key(self, value):
		if isinstance(value, float) and np.isnan(value):
			return "__nan__"

		return value

	def _compute_drop_idx(self):
		if self.drop is None:
			return [None] * len(self.categories_)

		if self.drop == "first":
			return [0] * len(self.categories_)

		if self.drop == "if_binary":
			result = []

			for categories in self.categories_:
				if len(categories) == 2:
					result.append(0)
				else:
					result.append(None)

			return result

		if len(self.drop) != len(self.categories_):
			raise ValueError(
				"drop array length must match number of features"
			)

		result = []

		for feature_idx, drop_value in enumerate(self.drop):
			categories = self.categories_[feature_idx]

			matches = []

			for idx, category in enumerate(categories):
				if self._values_equal(category, drop_value):
					matches.append(idx)

			if len(matches) == 0:
				raise ValueError(
					f"Drop category {drop_value!r} not found"
				)

			result.append(matches[0])

		return result

	def _values_equal(self, a, b):
		if isinstance(a, float) and isinstance(b, float):
			if np.isnan(a) and np.isnan(b):
				return True

		return a == b

	def _is_ignored(self, value):
		if self.ignore_categories is None:
			return False

		for ignore_val in self.ignore_categories:
			if self._values_equal(value, ignore_val):
				return True

		return False

	def _compute_feature_layout(self):
		feature_sizes = []

		for feature_idx, categories in enumerate(self.categories_):
			size = len(categories)

			if self.drop_idx_[feature_idx] is not None:
				size -= 1

			feature_sizes.append(size)

		self.feature_sizes_ = feature_sizes

		offsets = [0]

		for size in feature_sizes:
			offsets.append(offsets[-1] + size)

		self.feature_offsets_ = offsets
		self.n_output_features_ = offsets[-1]

	def __sklearn_is_fitted__(self):
		return hasattr(self, "categories_")

