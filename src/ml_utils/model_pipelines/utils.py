def _get_available_models(models, families=None):
	result = {
		"models": list(models.keys()),
	}

	if families is not None:
		result["families"] = list(families.keys())

	return result

