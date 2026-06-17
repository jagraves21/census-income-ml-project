import dataclasses

from .base import DatasetManager

@dataclasses.dataclass
class Census(DatasetManager):
	project_root: str = None
	dataset_name: str = dataclasses.field(init=False, default="Census")

	# --- Filenames ---
	DATA_FILE: str = dataclasses.field(init=False, default="census-bureau.data", repr=False)
	COLUMNS_FILE: str = dataclasses.field(init=False, default="census-bureau.columns", repr=False)


	def __post_init__(self):
		super().__post_init__()


	# --- Raw Data Loaders ---
	def load_data(self, verbose=False):
		data = self._load_raw_data(verbose=verbose)
		columns = self._load_column_names(verbose=verbose)
		data.columns = columns
		return data


	# --- Helpers ---
	def _load_raw_data(self, verbose=False):
		return self.load_raw_dataframe(
			self.DATA_FILE,
			header=None,
			index_col=None,
			verbose=verbose
		)

	def _load_column_names(self, verbose=False):
		columns_df = self.load_raw_dataframe(
			self.COLUMNS_FILE,
			header=None,
			index_col=None,
			verbose=verbose
		)
		return columns_df.iloc[:, 0].astype(str).values


	# --- Feature Groups ---
	def get_target_feature(self):
		return "label"
	
	def get_sample_weights_feature(self):
		return "weight"

	def get_demographic_features(self):
		return [
			"age",
			"race",
			"sex",
			"hispanic origin",
			"citizenship",
			"country of birth self",
			"country of birth father",
			"country of birth mother"
		]

	def get_education_and_schooling_features(self):
		return [
			"education",
			"enroll in edu inst last wk"
		]


	def get_employment_type_and_job_structure_features(self):
		return [
			"class of worker",
			"full or part time employment stat",
			"member of a labor union",
			"reason for unemployment",
			"own business or self employed",
			"num persons worked for employer",
			"detailed occupation recode",
			"major occupation code",
			"detailed industry recode",
			"major industry code"
		]
	
	def get_income_and_financial_indicator_features(self):
		return [
			"wage per hour",
			"capital gains",
			"capital losses",
			"dividends from stocks"
		]
	
	def get_work_history_features(self):
		return [
			"weeks worked in year"
		]


	def get_veteran_and_military_status_features(self):
		return [
			"fill inc questionnaire for veteran's admin",
			"veterans benefits"
		]

	def get_household_and_family_structure_features(self):
		return [
			"marital stat",
			"detailed household and family stat",
			"detailed household summary in household",
			"family members under 18",
			"tax filer stat"
		]

	def get_geography_and_migration_history_features(self):
		return [
			"region of previous residence",
			"state of previous residence",
			"migration code-change in msa",
			"migration code-change in reg",
			"migration code-move within reg",
			"live in this house 1 year ago",
			"migration prev res in sunbelt"
		]

	def get_temporal_identifier_featuers(self):
		return [
			"year"
		]

	def get_numeric_categorical_features(self):
		return [
			#"age",
			"own business or self employed",
			"num persons worked for employer",
			"detailed occupation recode",
			"major occupation code",
			"detailed industry recode",
			"major industry code"
			#"weeks worked in year",
			"veterans benefits"
		]
