import json
from pathlib import Path
from jsonschema import Draft202012Validator


def _get_schema_path():
	return Path(__file__).resolve().parent / "pipeline.schema.json"


def _load_schema():
	schema_path = _get_schema_path()

	if not schema_path.exists():
		raise FileNotFoundError("Pipeline schema not found at {}".format(schema_path))

	with open(schema_path, "r", encoding="utf-8") as f:
		return json.load(f)


def is_valid_pipeline(spec):
	schema = _load_schema()
	validator = Draft202012Validator(schema)
	return validator.is_valid(spec)


def validate_pipeline(spec):
	schema = _load_schema()
	validator = Draft202012Validator(schema)

	errors = sorted(
		validator.iter_errors(spec),
		key=lambda e: list(e.path)
	)

	if not errors:
		return

	msgs = []

	for err in errors:
		if err.path:
			path = ".".join(str(p) for p in err.path)
		else:
			path = "<root>"

		msgs.append("{}: {}".format(path, err.message))

	raise ValueError("Pipeline validation failed:\n" + "\n".join(msgs))

