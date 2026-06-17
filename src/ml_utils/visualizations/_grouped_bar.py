from itertools import cycle, islice

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ..utils import ensure_dict


def _normalize(color, count):
	default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

	if color is None:
		color = default_colors

	if isinstance(color, pd.Series):
		color = color.tolist()

	if isinstance(color, np.ndarray):
		color = color.tolist()

	if isinstance(color, str):
		try:
			color = plt.get_cmap(color)
		except ValueError:
			pass

	if isinstance(color, plt.Colormap):
		return color(np.linspace(0, 1, count))

	if isinstance(color, (list, tuple)):
		if len(color) == 0:
			return list(islice(cycle(default_colors), count))

		if len(color) >= count:
			return list(color[:count])

		return list(islice(cycle(color), count))

	return [color] * count


def _get_groups(df, group_column):
	return df[group_column].tolist()


def _build_value_matrix(
	df,
	group_column,
	label_column,
	data_column,
	group_labels,
	unique_labels
):
	group_index = {group: ii for ii, group in enumerate(group_labels)}
	label_index = {label: ii for ii, label in enumerate(unique_labels)}

	matrix = np.full((len(group_labels), len(unique_labels)), np.nan)

	for row in df.itertuples(index=False):
		group = getattr(row, group_column)
		label = getattr(row, label_column)
		value = getattr(row, data_column)

		if group in group_index and label in label_index:
			matrix[group_index[group], label_index[label]] = value

	return matrix


def _compute_group_offsets(n_slots):
	group_width = 0.8
	step = group_width / n_slots
	start = -group_width / 2 + step / 2
	return [start + ii * step for ii in range(n_slots)], step


def _build_label_color_map(labels, colors):
	unique_labels = list(dict.fromkeys(labels))
	palette = _normalize(colors, len(unique_labels))

	return {
		label: palette[ii]
		for ii, label in enumerate(unique_labels)
	}


def _plot_bars_core(
	ax,
	orientation,
	group_centers,
	values_matrix,
	labels,
	label_colors,
	bar_kwargs,
):
	n_groups, n_labels = values_matrix.shape

	offsets, bar_width = _compute_group_offsets(n_labels)

	seen = set()

	for group_idx, center in enumerate(group_centers):
		for label_idx in range(n_labels):
			val = values_matrix[group_idx, label_idx]

			if np.isnan(val):
				continue

			label = labels[label_idx]
			color = label_colors[label]

			pos = center + offsets[label_idx]

			kwargs = ensure_dict(bar_kwargs)
			kwargs.setdefault("color", color)

			if label not in seen:
				kwargs["label"] = label
				seen.add(label)

			if orientation == "v":
				ax.bar(pos, val, width=bar_width, **kwargs)
			else:
				ax.barh(pos, val, height=bar_width, **kwargs)


def _plot_errorbars_asymmetric(
	ax,
	orientation,
	group_centers,
	values_matrix,
	lower_matrix,
	upper_matrix,
	labels,
	label_colors,
	zorder,
	alpha,
	errorbar_kwargs,
):
	errorbar_kwargs = ensure_dict(errorbar_kwargs)
	errorbar_kwargs.setdefault("zorder", zorder)
	errorbar_kwargs.setdefault("alpha", alpha)
	errorbar_kwargs.setdefault("fmt", "none")

	n_groups, n_labels = values_matrix.shape
	offsets, _ = _compute_group_offsets(n_labels)

	for group_idx, center in enumerate(group_centers):
		for label_idx in range(n_labels):
			val = values_matrix[group_idx, label_idx]
			lower = lower_matrix[group_idx, label_idx]
			upper = upper_matrix[group_idx, label_idx]

			if np.isnan(val) or np.isnan(lower) or np.isnan(upper):
				continue

			label = labels[label_idx]
			color = label_colors[label]

			pos = center + offsets[label_idx]

			kwargs = ensure_dict(errorbar_kwargs)
			kwargs.setdefault("ecolor", color)

			if orientation == "v":
				ax.errorbar(pos, val, yerr=[[lower], [upper]], **kwargs)
			else:
				ax.errorbar(val, pos, xerr=[[lower], [upper]], **kwargs)


def _set_labels(
	ax,
	orientation,
	group_centers,
	group_labels,
	title,
	xlabel,
	ylabel,
	label_rotation,
	label_ha,
	label_side,
):
	if orientation == "v":
		ax.set_xticks(group_centers)
		ax.set_xticklabels(group_labels, rotation=label_rotation, ha=label_ha)
	else:
		ax.set_yticks(group_centers)
		ax.set_yticklabels(group_labels, rotation=label_rotation, ha=label_ha)

		if label_side == "right":
			ax.yaxis.tick_right()
			ax.yaxis.set_label_position("right")
		else:
			ax.yaxis.tick_left()
			ax.yaxis.set_label_position("left")

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)


def grouped_bar_plot_core(
	df,

	data_column,
	group_column,
	label_column,

	std_column=None,
	min_column=None,
	max_column=None,

	position_column=None,
	orientation="v",

	title=None,
	xlabel=None,
	ylabel=None,

	label_rotation=45,
	label_ha="right",
	label_side="left",

	colors=None,
	std_colors="red",
	minmax_colors="black",

	bar_kwargs=None,
	std_errorbar_kwargs=None,
	minmax_errorbar_kwargs=None,

	with_legend=True,

	ax=None
):
	if label_column is None:
		raise ValueError("grouped_bar_plot_core requires label_column")

	if orientation not in ("h", "v"):
		raise ValueError(f"Invalid orientation: {orientation}. Expected 'h' or 'v'.")

	groups = _get_groups(df, group_column)
	group_labels = list(dict.fromkeys(groups))

	labels = df[label_column].tolist()
	unique_labels = list(dict.fromkeys(labels))

	if position_column is None:
		group_centers = list(range(len(group_labels)))
	else:
		group_centers = df[position_column].tolist()

	values_matrix = _build_value_matrix(
		df,
		group_column,
		label_column,
		data_column,
		group_labels,
		unique_labels,
	)

	std_matrix = None
	min_matrix = None
	max_matrix = None

	if std_column is not None:
		std_matrix = _build_value_matrix(
			df,
			group_column,
			label_column,
			std_column,
			group_labels,
			unique_labels,
		)

	if min_column is not None:
		min_matrix = _build_value_matrix(
			df,
			group_column,
			label_column,
			min_column,
			group_labels,
			unique_labels,
		)

	if max_column is not None:
		max_matrix = _build_value_matrix(
			df,
			group_column,
			label_column,
			max_column,
			group_labels,
			unique_labels,
		)

	label_colors = _build_label_color_map(labels, colors)
	std_label_colors = _build_label_color_map(labels, std_colors)
	minmax_label_colors = _build_label_color_map(labels, minmax_colors)

	bar_kwargs = ensure_dict(bar_kwargs)
	std_errorbar_kwargs = ensure_dict(std_errorbar_kwargs)
	minmax_errorbar_kwargs = ensure_dict(minmax_errorbar_kwargs)

	_plot_bars_core(
		ax=ax,
		orientation=orientation,
		group_centers=group_centers,
		values_matrix=values_matrix,
		labels=unique_labels,
		label_colors=label_colors,
		bar_kwargs=bar_kwargs,
	)

	if std_matrix is not None:
		lower = std_matrix
		upper = std_matrix

		_plot_errorbars_asymmetric(
			ax=ax,
			orientation=orientation,
			group_centers=group_centers,
			values_matrix=values_matrix,
			lower_matrix=lower,
			upper_matrix=upper,
			labels=unique_labels,
			label_colors=std_label_colors,
			zorder=3,
			alpha=1.0,
			errorbar_kwargs=std_errorbar_kwargs,
		)

	if min_matrix is not None and max_matrix is not None:
		lower_matrix = np.where(
			np.isnan(values_matrix) | np.isnan(min_matrix),
			np.nan,
			values_matrix - min_matrix,
		)

		upper_matrix = np.where(
			np.isnan(values_matrix) | np.isnan(max_matrix),
			np.nan,
			max_matrix - values_matrix,
		)

		_plot_errorbars_asymmetric(
			ax=ax,
			orientation=orientation,
			group_centers=group_centers,
			values_matrix=values_matrix,
			lower_matrix=lower_matrix,
			upper_matrix=upper_matrix,
			labels=unique_labels,
			label_colors=minmax_label_colors,
			zorder=2,
			alpha=1.0,
			errorbar_kwargs=minmax_errorbar_kwargs,
		)

	_set_labels(
		ax=ax,
		orientation=orientation,
		group_centers=group_centers,
		group_labels=group_labels,
		title=title,
		xlabel=xlabel,
		ylabel=ylabel,
		label_rotation=label_rotation,
		label_ha=label_ha,
		label_side=label_side,
	)

	if with_legend:
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(
			handles=handles,
			labels=labels,
			loc="upper left",
			bbox_to_anchor=(1.02, 1.0),
			ncol=1,
			frameon=True,
			columnspacing=0.8,
			handletextpad=0.4,
			borderaxespad=0
		)

