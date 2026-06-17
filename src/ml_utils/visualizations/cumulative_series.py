from matplotlib import pyplot as plt

from .utils import with_ax


@with_ax
def plot_cumulative_series(
	series,
	highlight_labels=None,
	title="Cumulative Feature Coverage",
	xlabel="Featuers",
	ylabel="Cumulative Value",
	line_marker="o",
	color=None,
	highlighted_color="red",
	ax=None
):
	y = series.values
	labels = list(series.index)
	n = len(y)
	x = list(range(n))

	if color is None:
		color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

	if highlight_labels is None:
		highlight_labels = set()
	else:
		highlight_labels = set(highlight_labels)

	highlighted_x = [
		ii
		for ii, label in enumerate(labels)
		if label in highlight_labels
	]
	normal_x = [
		ii
		for ii, label in enumerate(labels)
		if label not in highlight_labels
	]

	ax.plot(x, y, color=color, zorder=0)

	ax.scatter(
		normal_x, [y[ii] for ii in normal_x],
		marker=line_marker, s=20, color=color, zorder=1
	)

	ax.scatter(
		highlighted_x, [y[ii] for ii in highlighted_x],
		marker=line_marker, s=20, color=highlighted_color, zorder=1
	)

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation=45, ha="right")

