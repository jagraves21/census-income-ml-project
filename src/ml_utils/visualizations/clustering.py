from collections.abc import Iterable
from itertools import cycle, islice
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .missing import _NIU_VALUES
from .utils import with_ax

from ..utils import (
	ensure_set,
	ensure_dict
)


def _ensure_color_dict(x, default):
	if isinstance(x, dict):
		x = dict(x)
		x.setdefault("default", default)
		return x

	return {"default": default if x is None else x}


def _get_feature_color(feature, color_map):
	return color_map.get(feature, color_map.get("default"))


def _normalize_color(color, count):
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
		return list(color(np.linspace(0, 1, count)))

	if isinstance(color, (list, tuple)):
		if len(color) == 0:
			return list(islice(cycle(default_colors), count))

		if len(color) >= count:
			return list(color[:count])

		return list(islice(cycle(color), count))

	return [color] * count


def _filter_by_clusters(values, cluster_assignments, clusters, weights=None):
	values = np.asarray(values)
	cluster_assignments = np.asarray(cluster_assignments)

	mask = ~pd.isna(values) & ~pd.isna(cluster_assignments)
	if weights is not None:
		mask &= ~pd.isna(weights)
	values = values[mask]
	cluster_assignments = cluster_assignments[mask]
	weights = weights if weights is None else weights[mask]

	if clusters is None:
		clusters = np.unique(cluster_assignments)
	else:
		clusters = np.asarray(clusters)
		mask = np.isin(cluster_assignments, clusters)
		values = values[mask]
		cluster_assignments = cluster_assignments[mask]
		weights = weights if weights is None else weights[mask]

	return values, cluster_assignments, clusters, weights


def _weighted_resample(values, weights, size):
	if len(values) == 0:
		return values

	values = np.asarray(values)
	weights = np.asarray(weights)

	prob = weights / np.sum(weights)
	rng = np.random.default_rng(0)
	idx = rng.choice(len(values), size=size, replace=True, p=prob)
	return values[idx]


def _compute_global(values, weights):
	if weights is None:
		return values
	return _weighted_resample(values, weights, len(values))


def _prepare_clusters(clusters, include_overall):
	clusters = list(clusters)
	if include_overall:
		return ["Overall"] + clusters
	return clusters


def _iqr_bounds(values, k):
	values = np.asarray(values, copy=False)

	if len(values) == 0:
		return None, None

	q1 = np.nanpercentile(values, 25)
	q3 = np.nanpercentile(values, 75)
	iqr = q3 - q1

	low = q1 - (k * iqr)
	high = q3 + (k * iqr)

	return low, high


def _iqr_clip(values, k):
	values = np.asarray(values)

	if len(values) == 0:
		return values

	low, high = _iqr_bounds(values, k)
	if low is None:
		return values

	return np.clip(values, low, high)


def _iqr_drop(values, k):
	values = np.asarray(values)

	if len(values) == 0:
		return values

	low, high = _iqr_bounds(values, k)
	if low is None:
		return values

	mask = (values >= low) & (values <= high)
	return values[mask]


def _iqr_process(values, k=1.5, mode="drop"):
	if mode == "clip":
		return _iqr_clip(values, k)
	elif mode == "drop":
		return _iqr_drop(values, k)
	else:
		raise ValueError("mode must be 'clip' or 'drop'")


@with_ax
def plot_cluster_violins(
	df,
	feature,
	cluster_assignments,
	weight_column=None,
	clusters=None,
	cluster_labels=None,
	cluster_colors="hsv",
	include_overall=True,
	clip_outliers=False,
	title=None,
	xlabel=None,
	ylabel=None,
	ax=None
):
	values = np.asarray(df[feature])
	cluster_assignments = np.asarray(cluster_assignments)
	cluster_labels = ensure_dict(cluster_labels)

	if weight_column is None:
		weights = None
	else:
		weights = np.asarray(df[weight_column])

	values, cluster_assignments, clusters, weights = _filter_by_clusters(
		values,
		cluster_assignments,
		clusters,
		weights
	)

	cluster_palette = _normalize_color(cluster_colors, len(clusters))
	cluster_color_map = dict(zip(clusters, cluster_palette))

	grouped_data = []
	labels = []
	colors = []

	if include_overall:
		global_values = _compute_global(values, weights)

		if clip_outliers:
			global_values = _iqr_process(global_values)

		if len(global_values) > 0:
			grouped_data.append(global_values)
			labels.append("Overall")
			colors.append("gray")

	for cluster in clusters:
		mask = cluster_assignments == cluster
		cluster_values = values[mask]
		cluster_weights = None if weights is None else weights[mask]

		if len(cluster_values) == 0:
			warnings.warn(f"Cluster {cluster} is empty", RuntimeWarning)
			continue

		if cluster_weights is None:
			plot_values = cluster_values
		else:
			plot_values = _weighted_resample(
				cluster_values,
				cluster_weights,
				len(cluster_values)
			)
		
		if clip_outliers:
			cluster_values = _iqr_clip(cluster_values)

		grouped_data.append(plot_values)
		labels.append(cluster_labels.get(cluster, cluster))
		colors.append(cluster_color_map[cluster])

	if len(grouped_data) == 0:
		raise ValueError("No data available to plot.")

	parts = ax.violinplot(
		grouped_data,
		showmeans=True,
		showmedians=True,
		showextrema=False
	)

	for color, body in zip(colors, parts["bodies"]):
		body.set_facecolor(color)
		body.set_alpha(0.7)
		body.set_edgecolor("black")

	parts["cmeans"].set_color("black")
	parts["cmeans"].set_linewidth(2)

	parts["cmedians"].set_color("black")
	parts["cmedians"].set_linewidth(1)

	ax.set_xticks(range(1, len(labels) + 1))
	ax.set_xticklabels(labels, rotation=45, ha="right")

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)


@with_ax
def plot_cluster_stacked_bars(
	df,
	feature,
	cluster_assignments,
	weight_column=None,
	niu_categories=None,
	clusters=None,
	cluster_labels=None,
	category_colors=None,
	include_overall=True,
	title=None,
	xlabel=None,
	ylabel=None,
	ax=None
):
	values = np.asarray(df[feature])
	cluster_assignments = np.asarray(cluster_assignments)
	cluster_labels = ensure_dict(cluster_labels)

	if weight_column is None:
		weights = None
	else:
		weights = np.asarray(df[weight_column])

	values, cluster_assignments, clusters, weights = _filter_by_clusters(
		values,
		cluster_assignments,
		clusters,
		weights
	)

	mask = ~pd.isna(values)
	values = values[mask]
	cluster_assignments = cluster_assignments[mask]
	weights = None if weights is None else weights[mask]

	categories = np.unique(values)
	categories = categories[~pd.isna(categories)]

	if niu_categories is None:
		niu_categories = set()
	elif niu_categories == "default":
		niu_categories = set(_NIU_VALUES)
	else:
		niu_categories = set(niu_categories)

	normal_categories = [
		value
		for value in categories
		if value not in niu_categories
	]
	niu_categories = [
		value
		for value in categories
		if value in niu_categories
	]
	categories = np.array(normal_categories + niu_categories)

	all_clusters = _prepare_clusters(clusters, include_overall)

	counts = np.zeros((len(all_clusters), len(categories)))

	if include_overall:
		for ii, value in enumerate(categories):
			mask = values == value
			if weights is None:
				counts[0, ii] = np.sum(mask)
			else:
				counts[0, ii] = np.sum(weights[mask])

	start_idx = 1 if include_overall else 0

	for ii, cluster in enumerate(clusters, start=start_idx):
		mask = cluster_assignments == cluster
		cluster_values = values[mask]
		cluster_weights = None if weights is None else weights[mask]

		for jj, value in enumerate(categories):
			mask = cluster_values == value
			if cluster_weights is None:
				counts[ii, jj] = np.sum(mask)
			else:
				counts[ii, jj] = np.sum(cluster_weights[mask])

	row_sums = counts.sum(axis=1, keepdims=True)
	row_sums[row_sums == 0] = 1
	proportions = counts / row_sums

	category_colors = _normalize_color(category_colors, len(categories))

	x = np.arange(len(all_clusters))
	bottom = np.zeros(len(all_clusters))

	for ii, category in enumerate(categories):
		if category in niu_categories:
			color = "black"
		else:
			color = category_colors[ii]

		ax.bar(
			x,
			proportions[:, ii],
			bottom=bottom,
			label=str(category),
			color=color
		)
		bottom += proportions[:, ii]

	ticklabels = ["Overall"] + [
		cluster_labels.get(cluster, cluster)
		for cluster in clusters
	] if include_overall else [
		cluster_labels.get(cluster, cluster)
		for cluster in clusters
	]

	ax.set_xticks(x)
	ax.set_xticklabels(ticklabels, rotation=45, ha="right")

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	ax.legend(
		title=feature,
		loc="upper left",
		bbox_to_anchor=(1.02, 1.0),
		ncol=1,
		frameon=True,
		columnspacing=0.8,
		handletextpad=0.4,
		borderaxespad=0
	)

def plot_cluster_features(
	df,
	cluster_assignments,
	features=None,
	weight_column=None,
	niu_categories=None,
	treat_as_catagorical=None,
	clusters=None,
	cluster_labels=None,
	include_overall=True,
	clip_outliers=False,
	bar_category_colors=list(plt.cm.tab20.colors),
	violin_cluster_colors="hsv",
	figsize=(6.4, 4.8),
	save_fun=None
):
	if features is None:
		features = df.columns.tolist()

	treat_as_catagorical = ensure_set(treat_as_catagorical)

	bar_category_colors = _ensure_color_dict(
		bar_category_colors,
		list(plt.cm.tab20.colors)
	)

	violin_cluster_colors = _ensure_color_dict(
		violin_cluster_colors,
		"hsv"
	)

	cluster_assignments = np.asarray(cluster_assignments)

	if clusters is None:
		clusters = np.unique(cluster_assignments)
	else:
		clusters = np.asarray(clusters)

	for feature in features:
		column = df[feature]

		if np.array_equal(column.values, cluster_assignments):
			continue

		if (
			pd.api.types.is_numeric_dtype(column)
			and feature not in treat_as_catagorical
		):
			_,fig,ax = plot_cluster_violins(
				df=df,
				feature=feature,
				cluster_assignments=cluster_assignments,
				weight_column=weight_column,
				clusters=clusters,
				cluster_labels=cluster_labels,
				cluster_colors=_get_feature_color(
					feature, violin_cluster_colors
				),
				include_overall=include_overall,
				clip_outliers=clip_outliers,
				title=f"{feature} Distribution by Cluster",
				xlabel="Cluster",
				ylabel=feature,
				figsize=figsize,
				return_fig=True
			)
			plt.show()
			if save_fun is not None:
				save_fun(fig, feature)
		else:
			_,fig,ax = plot_cluster_stacked_bars(
				df=df,
				feature=feature,
				cluster_assignments=cluster_assignments,
				weight_column=weight_column,
				niu_categories=niu_categories,
				clusters=clusters,
				cluster_labels=cluster_labels,
				category_colors=_get_feature_color(
					feature, bar_category_colors
				),
				include_overall=include_overall,
				title=f"{feature} Distribution by Cluster",
				xlabel="Cluster",
				ylabel=feature,
				figsize=figsize,
				return_fig=True
			)
			plt.show()
			if save_fun is not None:
				save_fun(fig, feature)

