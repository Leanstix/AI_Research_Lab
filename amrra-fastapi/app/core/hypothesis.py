# app/core/hypothesis.py
import os
from .llm import build_hypotheses_with_llm

def generate_hypotheses(question: str, dataset_meta: dict, sources: list = None) -> list:
    """
    Uses LLM when ENABLE_LLM=true, else returns deterministic hypotheses.
    """
    use_llm = (os.getenv("ENABLE_LLM","false").lower() == "true")
    if use_llm:
        try:
            data, _meta = build_hypotheses_with_llm(question, dataset_meta or {}, sources or [], max_items=3)
            hyps = data.get("hypotheses") or []
            if hyps: return hyps
        except Exception:
            pass  # fall back if provider fails

    # Fallback deterministic
    ds_name = (dataset_meta or {}).get("name","dataset")
    target = (dataset_meta or {}).get("target_name","target")
    base = [
        f"H0: No improvement over a naive baseline in predicting {target} on {ds_name}.",
        f"H1: A non-linear model yields higher ROC AUC than a linear baseline on {ds_name}.",
        f"H2: Feature normalization has limited impact on {ds_name}."
    ]
    if question:
        base[-1] = f"H2: '{question}' is supported under standard evaluation on {ds_name}."
    return base[:3]
