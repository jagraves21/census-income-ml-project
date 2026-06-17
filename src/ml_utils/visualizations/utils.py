from functools import wraps
from matplotlib import pyplot as plt


def with_ax(func):
	@wraps(func)
	def wrapper(
		*args,
		ax=None,
		figsize=(6.4, 4.8),
		return_fig=False,
		**kwargs
	):
		created = ax is None

		if created:
			fig, ax = plt.subplots(figsize=figsize)
		else:
			fig = ax.figure

		result = func(*args, ax=ax, **kwargs)

		if created and not return_fig:
			plt.show()

		if return_fig:
			return result, fig, ax

		return result

	return wrapper

