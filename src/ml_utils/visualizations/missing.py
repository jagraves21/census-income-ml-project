from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import (
	is_string_dtype,
	is_object_dtype,
	is_categorical_dtype
)

from .bar import bar_plot
from .utils import with_ax


_NIU_VALUES = [
	"Not in universe",
	"Not in universe or children",
	"Not in universe under 1 year old",
	"Children or Armed Forces",
]


@with_ax
def plot_missing_values(
	df,
	weight_column=None,
	only_missing=False,
	horizontal=True,
	ax=None
):
	if weight_column is not None:
		w = df[weight_column].to_numpy()
		w = w / np.sum(w)

		missing_df = (
			df
			.isnull()
			.astype(float)
			.multiply(w, axis=0)
			.sum()
			.sort_values(ascending=horizontal)
			.to_frame(name="Percentage (%)")
		)
		xlabel = "Weighted Proportion"
	else:
		missing_df = (
			df
			.isnull()
			.mean()
			.sort_values(ascending=not horizontal)
			.to_frame(name="Percentage (%)")
		)
		xlabel = "Proportion"

	if only_missing:
		missing_df = missing_df[missing_df["Percentage (%)"] > 0]
	
	title = "Missing Values Proportion by Feature"
	if weight_column is not None:
		title += " (weighted)"

	bar_plot(
		df=missing_df,
		data_column="Percentage (%)",
		colors=None,
		title=title,
		xlabel=xlabel,
		ylabel="Feature",
		horizontal=horizontal,
		ax=ax,
	)

	return (
		missing_df
		.sort_values("Percentage (%)", ascending=False)
		.reset_index(names="Feature")
		.assign(**{
			"Percentage (%)": lambda df: df["Percentage (%)"] * 100
		})
	)


@with_ax
def plot_not_in_universe(
	df,
	niu_values=None,
	weight_column=None,
	niu_only=False,
	horizontal=True,
	ax=None
):
	if niu_values is None:
		niu_values = list(_NIU_VALUES)

	niu_proportions = dict()
	niu_indicator_values = dict()

	for column in df.columns:
		if not (
			is_string_dtype(df[column])
			or is_object_dtype(df[column])
			or is_categorical_dtype(df[column])
		):
			continue

		mask = df[column].isin(niu_values)

		if weight_column is None:
			proportion = mask.mean()
		else:
			total_weight = df[weight_column].sum()
			proportion = (
				df.loc[mask, weight_column].sum() / total_weight
			)

		if proportion > 0 or not niu_only:
			niu_proportions[column] = proportion
			niu_indicator_values[column] = ", ".join(
				list(df.loc[mask, column].unique())
			)

	niu_df = (
		pd.DataFrame([
			pd.Series(niu_indicator_values, name="NIU Indicator Value"),
			pd.Series(niu_proportions, name="Percentage (%)"),
		])
		.transpose()
		.sort_values("Percentage (%)", ascending=horizontal)
	)

	if niu_df.empty:
		print("No 'Not in Universe' values found.")
		return

	title = "Not in Universe (NIU) Proportion by Feature"
	if weight_column is not None:
		title += " (weighted)"

	bar_plot(
		df=niu_df,
		data_column="Percentage (%)",
		colors=None,
		title=title,
		xlabel="Weighted Proportion" if weight_column else "Proportion",
		ylabel="Feature",
		horizontal=horizontal,
		ax=ax,
	)

	niu_values = set(_NIU_VALUES)

	return (
		niu_df
		.sort_values("Percentage (%)", ascending=False)
		.reset_index(names="Feature")
		.assign(**{
			"Percentage (%)": lambda df: df["Percentage (%)"] * 100
		})
	)

