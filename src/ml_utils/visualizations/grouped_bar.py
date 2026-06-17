from ._grouped_bar import grouped_bar_plot_core

from .utils import with_ax


@with_ax
def va_grouped_bar_plot(
	label_rotation=45,
	**kwargs
):
	kwargs["label_rotation"] = label_rotation
	kwargs["orientation"] = "v"
	return grouped_bar_plot_core(**kwargs)


@with_ax
def ha_grouped_bar_plot(
	label_rotation=0,
	**kwargs
):
	kwargs["label_rotation"] = label_rotation
	kwargs["orientation"] = "h"
	return grouped_bar_plot_core(**kwargs)


@with_ax
def grouped_bar_plot(horizontal=False, **kwargs):
	if horizontal:
		return ha_grouped_bar_plot(**kwargs)
	return va_grouped_bar_plot(**kwargs)
