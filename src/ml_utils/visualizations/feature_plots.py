from IPython.display import display, HTML

from .distributions import (
	plot_numeric_distribution,
	plot_categorical_distribution
)

from .target_analysis import (
	plot_numeric_vs_target,
	plot_categorical_vs_target
)

from ..utils import (
	ensure_set,
	ensure_dict
)


def plot_column(
	df,
	column,
	target,
	weight_column,
	numeric_categorical_columns,
	dist_kwargs=None,
	target_kwargs=None
):
	dist_kwargs = ensure_dict(dist_kwargs)
	target_kwargs = ensure_dict(target_kwargs)

	if (
		df[column].dtype in ["int64", "float64"]
		and not numeric_categorical_columns
	):
		"""plot_numeric_distribution(
			df,
			column,
			weight_column=weight_column,
			**dist_kwargs
		)"""

		plot_numeric_vs_target(
			df,
			column,
			target,
			weight_column=weight_column,
			**target_kwargs
		)
	else:
		"""plot_categorical_distribution(
			df,
			column,
			weight_column=weight_column,
			**dist_kwargs
		)"""

		plot_categorical_vs_target(
			df,
			column,
			target=target,
			weight_column=weight_column,
			**target_kwargs
		)


def plot_feature_group(
	df,
	feature_list,
	target="label",
	weight_column="weight",
	numeric_categorical_columns=None,
	title=None,
	plot_kwargs=None
):
	numeric_categorical_columns = ensure_set(numeric_categorical_columns)
	plot_kwargs = ensure_dict(plot_kwargs)

	global_kwargs = plot_kwargs.get("_global", {"figsize": (12.8, 7.2)})

	if title:
		display(HTML(f"<h1>{title}</h1>"))

	for column in feature_list:
		if column not in df.columns:
			display(HTML(
				f"<h3 style=\"color: red;\">{column} column not found</h3>"
			))
			continue

		display(HTML(f"<h3>{column}</h3>"))

		column_cfg = plot_kwargs.get(column, {})

		dist_overrides = column_cfg.get("_dist", {})
		target_overrides = column_cfg.get("_target", {})

		dist_kwargs = {**global_kwargs, **dist_overrides}
		target_kwargs = {**global_kwargs, **target_overrides}

		plot_column(
			df,
			column,
			target,
			weight_column=weight_column,
			numeric_categorical_columns=column in numeric_categorical_columns,
			dist_kwargs=dist_kwargs,
			target_kwargs=target_kwargs
		)

