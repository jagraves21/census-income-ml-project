from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_matrix_heatmap(
	ax,
	matrix,
	cmap="viridis",
	vmin=None,
	vmax=None,
	title=None,
	cbar_label=None,
	show_values=False,
	fmt="{:.2f}",
	fontsize=8,
	text_color_threshold=None,
	xtick_rotation=45,
	ytick_rotation=0,
	annot_kwargs=None,
	grid=False,
	linewidths=0,
	linecolor="white",
	aspect="auto"
):
	annot_kwargs = annot_kwargs or {}

	im = ax.imshow(
		matrix.values,
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		aspect=aspect
	)

	ax.set_xticks(range(len(matrix.columns)))
	ax.set_xticklabels(matrix.columns, rotation=xtick_rotation, ha="right")

	ax.set_yticks(range(len(matrix.index)))
	ax.set_yticklabels(matrix.index, rotation=ytick_rotation)

	if title:
		ax.set_title(title)

	if grid:
		ax.set_xticks([x - 0.5 for x in range(1, len(matrix.columns))], minor=True)
		ax.set_yticks([y - 0.5 for y in range(1, len(matrix.index))], minor=True)
		ax.grid(which="minor", color=linecolor, linestyle="-", linewidth=linewidths)
		ax.tick_params(which="minor", bottom=False, left=False)

	if show_values:
		for row in range(matrix.shape[0]):
			for column in range(matrix.shape[1]):
				val = matrix.values[row, column]

				if text_color_threshold is None:
					text_color = "black"
				else:
					text_color = "white" if abs(val) > text_color_threshold else "black"

				ax.text(
					column,
					row,
					fmt.format(val),
					ha="center",
					va="center",
					color=text_color,
					fontsize=fontsize,
					**annot_kwargs
				)

	if cbar_label:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)

		cbar = ax.figure.colorbar(im, cax=cax)
		cbar.set_label(cbar_label)

