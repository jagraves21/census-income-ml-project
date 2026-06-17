import numpy as np

from ..mlpipelines import build_pipeline


_NIU_VALUES = [
	"Not in universe",
	"Not in universe or children",
	"Not in universe under 1 year old",
	"Children or Armed Forces",
]


def _get_categorical_transformer(categorical_columns, remove_niu=True):
	if remove_niu:
		ignore_categories = _NIU_VALUES + [np.nan]
	else:
		ignore_categories = [np.nan]

	return {
		"name": "categorical",
		"columns": categorical_columns,
		"transformer": {
			"name": "OneHotEncoderWithIgnore",
			"params": {
				"ignore_categories": ignore_categories,
				"sparse_output": False,
				"handle_unknown": "ignore"
			}
		}
	}


def _get_numeric_transformer(numeric_columns):
	return {
		"name": "numeric",
		"columns": numeric_columns,
		"transformer": {
			"name": "Pipeline",
			"steps": [
				{
					"name": "SimpleImputer",
					"params": {
						"strategy": "median"
					}
				},
				{
					"name": "StandardScaler",
				}
			]
		}
	}

	return spec


def _get_preprocessing_spec(
	categorical_columns,
	numeric_columns,
	remove_niu=True
):
	return {
		"name": "preprocessing",
		"transformers": [
			_get_categorical_transformer(
				categorical_columns, remove_niu=remove_niu
			),
			_get_numeric_transformer(numeric_columns)
		]
	}


def get_preprocessing_pipeline(
	categorical_columns,
	numeric_columns,
	remove_niu=True
):
	return build_pipeline(
		_get_preprocessing_spec(
			categorical_columns, numeric_columns, remove_niu
		)
	)

