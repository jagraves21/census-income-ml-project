def get_categorical_numeric_split(df, treat_as_categorical=None):
	categorical_set = set(
		df.select_dtypes(include=["object", "category", "bool"]).columns
	) | set(treat_as_categorical or [])

	categorical_columns = [
		column for column in df.columns
		if column in categorical_set
	]

	numeric_columns = [
		column for column in df.columns
		if column not in categorical_set
	]

	return categorical_columns, numeric_columns

