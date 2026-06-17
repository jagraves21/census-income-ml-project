from .hdbscan import register_hdbscan
from .umap import register_umap
from .xgboost import register_xgboost


def load_third_party_estimators():
	register_hdbscan()
	register_umap()
	register_xgboost()

