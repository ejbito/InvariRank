from ..retrieval import LightGCNRetriever
from .config_utils import cfg_get


def build_retriever(cfg):
    method = str(cfg_get(cfg, "retrieval.method", "lightgcn")).lower()
    if method in {"lightgcn", "lgcn"}:
        return LightGCNRetriever(cfg)
    raise ValueError(f"Unsupported retrieval method: {method}")
