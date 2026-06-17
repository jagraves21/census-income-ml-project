from IPython.display import display, HTML

from ..visualizations import bar_plot


def display_importance_summary(
	df,
	importance_column,
	sign_column=None,
	top_n=10,
	split_positive_negative=False
):
	# the sign_column determines direction (positive vs negative effect)
	# the importance_column measures magnitude of impact
	# we do NOT sort by sign because it may not be magnitude
	# we split by sign first, then rank within each group by importance
	# this avoids mixing direction while still showing strongest drivers
	if split_positive_negative:
		if sign_column is None:
			sign_column = importance_column

		top_pos = df[df[sign_column] > 0].sort_values(
			importance_column, ascending=False
		).head(top_n)

		top_neg = df[df[sign_column] < 0].sort_values(
			importance_column, ascending=False
		).head(top_n)

		display(HTML("<h4>Top Positive Drivers</h4>"))
		display(top_pos)

		display(HTML("<h4>Top Negative Drivers</h4>"))
		display(top_neg)
	else:
		top_df = df.sort_values(
			importance_column, ascending=False
		).head(top_n)

		display(HTML("<h4>Top Important Features</h4>"))
		display(top_df)

	display(HTML("<h4>Feature Importance by Type</h4>"))
	group_type = (
		df
		.groupby("feature_type", as_index=False)[importance_column]
		.sum()
		.sort_values(importance_column, ascending=False)
		.rename(columns={
			"feature_type": "feature_group",
			importance_column: f"mean_{importance_column}"
		})
	)
	display(group_type)

	display(HTML("<h4>Feature Importance by Type + Name</h4>"))
	group_type_name = (
		df
		.groupby(
			["feature_type", "feature_name"], as_index=False
		)[importance_column]
		.mean()
		.sort_values(importance_column, ascending=False)
		.rename(columns={
			importance_column: f"mean_{importance_column}"
		})
		.reset_index(drop=True)
	)
	display(group_type_name)
	
