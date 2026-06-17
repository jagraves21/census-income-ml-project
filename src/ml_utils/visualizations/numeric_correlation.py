from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

from ._correlation import plot_matrix_heatmap
from .utils import with_ax

from ..utils import ensure_dict


def weighted_corr_matrix(df, weights):
	df = df.copy()
	w = np.asarray(weights)

	if len(w) != len(df):
		raise ValueError("weights must match number of rows in df")

	x = df.to_numpy(dtype=float)
	w = w / np.sum(w)

	mean = np.sum(x * w[:, None], axis=0)
	x = x - mean

	cov = (x * w[:, None]).T @ x
	var = np.diag(cov)

	std = np.sqrt(var)
	std[std == 0] = np.nan

	corr = cov / np.outer(std, std)

	return pd.DataFrame(corr, index=df.columns, columns=df.columns)


@with_ax
def plot_correlation_heatmap(
	df,
	columns=None,
	weight_column=None,
	treat_as_categorical=None,
	ax=None,

	cmap="coolwarm",
	vmin=-1,
	vmax=1,
	show_values=False,
	fmt="{:.2f}",
	fontsize=8,
	text_color_threshold=0.75,
	xtick_rotation=45,
	ytick_rotation=0,
	title="Correlation Heatmap (Pearson)",
	cbar_label="Pearson Correlation",
	aspect="equal",
	heatmap_kwargs=None
):
	heatmap_kwargs = ensure_dict(heatmap_kwargs)

	if columns is not None:
		columns = columns(df)
	else:
		numeric_df = df.select_dtypes(include="number")

		if treat_as_categorical is not None:
			numeric_df = numeric_df.drop(
				columns=treat_as_categorical,
				errors="ignore"
			)

		if weight_column is not None:
			numeric_df = numeric_df.drop(
				columns=[weight_column], errors="ignore"
			)

		columns = numeric_df.columns

	sub_df = df[columns]

	if weight_column is not None:
		weights = df[weight_column]
		corr = weighted_corr_matrix(sub_df, weights)
	else:
		corr = sub_df.corr()

	plot_matrix_heatmap(
		ax=ax,
		matrix=corr,
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

	return corr

