from matplotlib import pyplot as plt

from .bar import bar_plot
from .utils import with_ax


@with_ax
def plot_summary_statistics(
	df,

	mean_column="mean",
	x_label_column=None,
	std_column="std",
	min_column="min",
	max_column="max",

	title="Summary Statistics",
	xlabel=None,
	ylabel=None,

	mean_color="tab:blue",
	mean_alpha=0.5,

	std_alpha=1.0,
	std_capsize=None,
	std_linewidth=5,

	minmax_color="black",
	minmax_alpha=1.0,
	minmax_capsize=None,
	minmax_linewidth=1,

	highlight_labels=None,
	highlighted_color="tab:orange",

	ax=None,
	horizontal=False,
):
	colors = mean_color
	if highlight_labels is not None:
		colors = [
			highlighted_color if lbl in highlight_labels else mean_color
			for lbl in (df[x_label_column].tolist()
			if x_label_column else df.index)
		]

	return bar_plot(
		df=df,
		data_column=mean_column,
		label_column=x_label_column,
		std_column=std_column,
		min_column=min_column,
		max_column=max_column,

		position_column=None,

		xlabel=xlabel,
		ylabel=ylabel,

		label_rotation=45,
		label_ha="right",
		label_side="left",

		colors=colors,
		std_colors=colors,
		minmax_colors=minmax_color,

		bar_kwargs={"alpha":mean_alpha},
		std_errorbar_kwargs={
			"alpha":mean_alpha,
			"linewidth": std_linewidth,
			"capsize": std_capsize 
		},
		minmax_errorbar_kwargs={
			"alpha": minmax_alpha,
			"linewidth": minmax_linewidth,
			"capsize": minmax_capsize 
		},

		ax=ax,

		horizontal=horizontal
	)

