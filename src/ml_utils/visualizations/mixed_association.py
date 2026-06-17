from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

from ._correlation import plot_matrix_heatmap

from .utils import with_ax

from ..utils import (
	ensure_dict,
	get_categorical_numeric_split
)


def eta_squared(categorical, numeric, weights=None):
	data = pd.DataFrame({
		"categorical": categorical,
		"numeric": numeric
	})

	if weights is not None:
		data["weight"] = weights
	else:
		data["weight"] = 1.0

	data = data.dropna()

	if data.empty:
		return 0.0

	total_weight = data["weight"].sum()

	if total_weight == 0:
		return 0.0

	global_mean = np.average(
		data["numeric"],
		weights=data["weight"]
	)

	total_var = np.average(
		(data["numeric"] - global_mean) ** 2,
		weights=data["weight"]
	)

	if total_var == 0:
		return 0.0

	between_var = 0.0

	for _, g in data.groupby("categorical"):
		group_weight = g["weight"].sum()

		if group_weight == 0:
			continue

		group_mean = np.average(
			g["numeric"],
			weights=g["weight"]
		)

		between_var += group_weight * (group_mean - global_mean) ** 2

	between_var /= total_weight

	return between_var / total_var


def eta_squared_matrix(
	df,
	categorical_columns,
	numeric_columns,
	weight_column=None
):
	df = df.copy()

	weights = df[weight_column] if weight_column is not None else None

	matrix = pd.DataFrame(
		index=categorical_columns,
		columns=numeric_columns,
		dtype=float
	)

	for c in categorical_columns:
		for v in numeric_columns:
			matrix.loc[c, v] = eta_squared(
				df[c],
				df[v],
				weights=weights
			)

	return matrix


@with_ax
@with_ax
def plot_mixed_association_heatmap(
	df,
	columns=None,
	weight_column=None,
	treat_as_categorical=None,
	ax=None,

	cmap="Reds",
	vmin=0,
	vmax=1,
	show_values=False,
	fmt="{:.2f}",
	fontsize=8,
	text_color_threshold=0.75,
	xtick_rotation=45,
	ytick_rotation=0,
	title="Categorical vs Numeric Association (η²)",
	cbar_label="Explained Variance (η²)",
	aspect="equal",
	heatmap_kwargs=None
):
	heatmap_kwargs = ensure_dict(heatmap_kwargs)

	df = df.copy()

	if columns is not None:
		categorical_columns, numeric_columns = columns(df)
	else:
		categorical_columns, numeric_columns = get_categorical_numeric_split(
			df, treat_as_categorical
		)

	if weight_column is not None and weight_column in numeric_columns:
		numeric_columns.remove(weight_column)

	matrix = eta_squared_matrix(
		df,
		categorical_columns,
		numeric_columns,
		weight_column=weight_column
	)

	if matrix.empty:
		raise ValueError("No categorical vs numeric relationships found.")

	plot_matrix_heatmap(
		ax=ax,
		matrix=matrix,
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		title=title,
		cbar_label=cbar_label,
		show_values=show_values,
		fmt=fmt,
		fontsize=fontsize,
		text_color_threshold=text_color_threshold,
		xtick_rotation=xtick_rotation,
		ytick_rotation=ytick_rotation,
		aspect=aspect,
		**heatmap_kwargs
	)

