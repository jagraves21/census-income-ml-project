from itertools import cycle, islice

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ..utils import ensure_dict


def _normalize_color(color, count):
	# plt.rcParams["axes.prop_cycle"] now returns RGB(A) tuples instead of hex
	# strings in newer Matplotlib versions, so we can no longer rely on this
	#default_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
	default_color = "tab:blue"

    if color is None:
        color = default_color

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
            return [default_color] * count

        if len(color) >= count:
            return list(color[:count])

        return list(islice(cycle(color), count))

    return [color] * count


def _get_series(df, column):
	return None if column is None else df[column].tolist()


def _set_labels(
	ax,
	positions,
	labels,
	orientation,
	title,
	xlabel,
	ylabel,
	label_rotation,
	label_ha,
	label_side,
):
	if orientation == "v":
		ax.set_xticks(positions)
		ax.set_xticklabels(labels, rotation=label_rotation, ha=label_ha)
	else:
		ax.set_yticks(positions)
		ax.set_yticklabels(labels, rotation=label_rotation, ha=label_ha)

		if label_side == "right":
			ax.yaxis.tick_right()
			ax.yaxis.set_label_position("right")
		else:
			ax.yaxis.tick_left()
			ax.yaxis.set_label_position("left")

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)


def _plot_bars_core(
	ax,
	orientation,
	positions,
	values,
	colors,
	bar_width,
	zorder,
	alpha,
	bar_kwargs,
):
	bar_kwargs = ensure_dict(bar_kwargs)
	bar_kwargs.setdefault("zorder", zorder)
	bar_kwargs.setdefault("alpha", alpha)

	for ii in range(len(positions)):
		kwargs = ensure_dict(bar_kwargs)
		kwargs.setdefault("color", colors[ii])

		if orientation == "v":
			ax.bar(positions[ii], values[ii], width=bar_width, **kwargs)
		else:
			ax.barh(positions[ii], values[ii], height=bar_width, **kwargs)


def _plot_errorbars(
	ax,
	orientation,
	positions,
	values,
	lower_errors,
	upper_errors,
	colors,
	zorder,
	alpha,
	errorbar_kwargs,
):
	errorbar_kwargs = ensure_dict(errorbar_kwargs)
	errorbar_kwargs.setdefault("zorder", zorder)
	errorbar_kwargs.setdefault("alpha", alpha)
	errorbar_kwargs.setdefault("fmt", "none")

	for ii in range(len(positions)):
		if (
			values[ii] is None
			or lower_errors[ii] is None
			or upper_errors[ii] is None
		):
			continue

		kwargs = ensure_dict(errorbar_kwargs)
		kwargs.setdefault("ecolor", colors[ii])
			
		if orientation == "v":
			ax.errorbar(
				positions[ii],
				values[ii],
				yerr=[[lower_errors[ii]], [upper_errors[ii]]],
				**kwargs,
			)
		else:
			ax.errorbar(
				values[ii],
				positions[ii],
				xerr=[[lower_errors[ii]], [upper_errors[ii]]],
				**kwargs,
			)


def bar_plot_core(
	df,
	data_column,
	label_column=None,

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
	
	ax=None
):
	if orientation not in ("h", "v"):
		raise ValueError(
			f"Invalid orientation: {orientation}. Expected 'h' or 'v'."
		)

	n_rows = len(df)

	if label_column is None:
		labels = df.index.tolist()
	else:
		labels = df[label_column].tolist()

	if position_column is None:
		positions = list(range(n_rows))
	else:
		positions = df[position_column].tolist()

	values = _get_series(df, data_column)
	stds = _get_series(df, std_column)
	mins = _get_series(df, min_column)
	maxs = _get_series(df, max_column)

	colors = _normalize_color(colors, n_rows)
	std_colors = _normalize_color(std_colors, n_rows)
	minmax_colors = _normalize_color(minmax_colors, n_rows)

	#bar_kwargs = ensure_dict(bar_kwargs)
	#std_errorbar_kwargs = ensure_dict(std_errorbar_kwargs)
	#minmax_errorbar_kwargs = ensure_dict(minmax_errorbar_kwargs)

	bar_width = 0.8

	_plot_bars_core(
		ax=ax,
		orientation=orientation,
		positions=positions,
		values=values,
		colors=colors,
		bar_width=bar_width,
		zorder=1,
		alpha=1.0,
		bar_kwargs=bar_kwargs,
	)

	if stds is not None:
		lower = stds
		upper = stds

		_plot_errorbars(
			ax=ax,
			orientation=orientation,
			positions=positions,
			values=values,
			lower_errors=lower,
			upper_errors=upper,
			colors=std_colors,
			zorder=3,
			alpha=1.0,
			errorbar_kwargs=std_errorbar_kwargs,
		)

	if mins is not None and maxs is not None:
		lower = [
			values[ii] - mins[ii] if values[ii] is not None else None
			for ii in range(n_rows)
		]
		upper = [
			maxs[ii] - values[ii] if values[ii] is not None else None
			for ii in range(n_rows)
		]

		_plot_errorbars(
			ax=ax,
			orientation=orientation,
			positions=positions,
			values=values,
			lower_errors=lower,
			upper_errors=upper,
			colors=minmax_colors,
			zorder=2,
			alpha=1.0,
			errorbar_kwargs=minmax_errorbar_kwargs,
		)

	_set_labels(
		ax=ax,
		positions=positions,
		labels=labels,
		orientation=orientation,
		title=title,
		xlabel=xlabel,
		ylabel=ylabel,
		label_rotation=label_rotation,
		label_ha=label_ha,
		label_side=label_side,
	)

