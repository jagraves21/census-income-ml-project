from ._bar import bar_plot_core

from .utils import with_ax

from ..utils import ensure_dict


@with_ax
def va_bar_plot(
	label_rotation=45,
	**kwargs
):
	kwargs["label_rotation"] = label_rotation
	kwargs["orientation"] = "v"
	return bar_plot_core(**kwargs)


@with_ax
def ha_bar_plot(
	label_rotation=0,
	**kwargs
):
	kwargs["label_rotation"] = label_rotation
	kwargs["orientation"] = "h"
	return bar_plot_core(**kwargs)


@with_ax
def bar_plot(horizontal=False, **kwargs):
	if horizontal:
		return ha_bar_plot(**kwargs)
	return va_bar_plot(**kwargs)

