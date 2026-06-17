import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._correlation import plot_matrix_heatmap
from .utils import with_ax

from ..utils import ensure_dict


def cramers_v(x, y, weights=None):
	if weights is None:
		contingency_table = pd.crosstab(
			x.fillna("<<NA>>"),
			y.fillna("<<NA>>")
		)
	else:
		temp_df = pd.DataFrame({
			"x": x.fillna("<<NA>>"),
			"y": y.fillna("<<NA>>"),
			"w": weights
		})

		contingency_table = temp_df.pivot_table(
			index="x",
			columns="y",
			values="w",
			aggfunc="sum",
			fill_value=0
		)

	if contingency_table.size == 0:
		return 0.0

	chi2_statistic = chi2_contingency(contingency_table)[0]
	sample_size = contingency_table.to_numpy().sum()

	if sample_size == 0:
		return 0.0

	phi_squared = chi2_statistic / sample_size

	num_rows, num_columns = contingency_table.shape

	phi_squared_corrected = max(
		0,
		phi_squared - ((num_columns - 1) * (num_rows - 1)) / max(sample_size - 1, 1)
	)

	row_correction = num_rows - ((num_rows - 1) ** 2) / max(sample_size - 1, 1)
	col_correction = num_columns - ((num_columns - 1) ** 2) / max(sample_size - 1, 1)

	degrees_of_freedom = min(
		col_correction - 1,
		row_correction - 1
	)

	degrees_of_freedom = max(degrees_of_freedom, 1e-12)

	return np.sqrt(phi_squared_corrected / degrees_of_freedom)


def cramers_v_matrix(df, weight_column=None):
	columns = df.columns.to_list()
	if weight_column is not None and weight_column in df.columns:
		columns.remove(weight_column)
	
	matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

	weights = df[weight_column] if weight_column is not None else None

	for ii, c1 in enumerate(columns):
		for jj, c2 in enumerate(columns):
			if jj < ii:
				continue
			elif ii == jj:
				matrix.loc[c1, c2] = 1.0
			else:
				val = cramers_v(
					df[c1],
					df[c2],
					weights=weights
				)
				matrix.loc[c1, c2] = val
				matrix.loc[c2, c1] = val

	return matrix


@with_ax
@with_ax
def plot_cramers_v_heatmap(
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
	title="Association Heatmap (Cramér's V)",
	cbar_label="Cramér's V",
	aspect="equal",
	heatmap_kwargs=None
):
	heatmap_kwargs = ensure_dict(heatmap_kwargs)

	if columns is not None:
		columns = columns(df)
	else:
		categorical_set = set(
			df.select_dtypes(exclude="number").columns
		)

		categorical_set.update(treat_as_categorical or [])

		if weight_column is not None:
			categorical_set.add(weight_column)

		columns = [
			column
			for column in df.columns
			if column in categorical_set
		]

	corr = cramers_v_matrix(df[columns], weight_column=weight_column)

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

