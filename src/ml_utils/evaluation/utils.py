def long_results_to_wides(df):
	stat_cols = [
		column
		for column in df.columns
		if column not in ["model", "metric"]
	]

	melted = df.melt(
		id_vars=["model", "metric"],
		value_vars=stat_cols,
		var_name="stat",
		value_name="value"
	)

	wide_df = melted.pivot_table(
		index=["metric", "stat"],
		columns="model",
		values="value"
	).sort_index()

	wide_df.columns.name = "model"
	return wide_df.reindex(df["metric"].unique(), level="metric")

