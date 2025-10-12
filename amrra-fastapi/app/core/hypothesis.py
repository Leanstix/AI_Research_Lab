import random

def generate_hypotheses(question: str, dataset_meta: dict) -> list:
    """
    Lightweight, deterministic-ish hypothesis generator.
    Later you can plug an LLM here (OpenAI, local transformers, etc.).
    """
    random.seed(42)
    ds_name = dataset_meta.get("name", "dataset")
    target = dataset_meta.get("target_name", "target")

    base = [
        f"H0: There is no improvement over a naive baseline in predicting {target} on {ds_name}.",
        f"H1: A non-linear model (Random Forest) yields higher ROC AUC than Logistic Regression on {ds_name}.",
        f"H2: Feature normalization minimally impacts model performance on {ds_name}.",
    ]

    # If a question is asked, steer one hypothesis
    if question:
        base.append(f"H3: With features available in {ds_name}, the claim — '{question.strip()}' — holds with statistically significant improvement over baseline.")
    return base[:3]  # keep to three for now

# Optional: plug-in for a real LLM later
def generate_hypotheses_llm(question: str, dataset_meta: dict, prompt_fn=None):
    """
    Placeholder: if you wire an LLM, call it here and return a list[str].
    """
    return generate_hypotheses(question, dataset_meta)
