# app/core/llm.py
import os, re, json, time
from typing import Any, Dict, Optional, Tuple, List
from jsonschema import validate, ValidationError

# Lazy imports so the app runs even if a provider isn't installed
def _lazy_openai():
    from openai import OpenAI
    return OpenAI

def _lazy_anthropic():
    from anthropic import Anthropic
    return Anthropic

JSON_SCHEMA_HYPOTHESES = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "minItems": 1, "maxItems": 6,
            "items": {"type": "string", "minLength": 10}
        }
    },
    "required": ["hypotheses"],
    "additionalProperties": False
}

JSON_SCHEMA_SUMMARIES = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "tldr": {"type": "string", "minLength": 8}
                },
                "required": ["title","url","tldr"],
                "additionalProperties": False
            }
        }
    },
    "required": ["summaries"],
    "additionalProperties": False
}

JSON_SCHEMA_CONCLUSION = {
    "type": "object",
    "properties": {
        "conclusion": {"type": "string", "minLength": 200},
        "strength_of_evidence": {"type": "string", "enum": ["strong","moderate","weak","inconclusive"]},
        "limitations": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": {"type": "string", "minLength": 10}
        },
        "next_steps": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": {"type": "string", "minLength": 8}
        }
    },
    "required": ["conclusion","strength_of_evidence","limitations","next_steps"],
    "additionalProperties": False
}

JSON_SCHEMA_STUDY = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array", "minItems": 1, "maxItems": 6,
            "items": {"type": "string", "minLength": 12}
        },
        "power_analysis": {
            "type": "object",
            "properties": {
                "assumptions": {
                    "type": "object",
                    "properties": {
                        "primary_metric": {"type": "string"},
                        "effect_size": {"type": ["number", "null"]},
                        "alpha": {"type": ["number","null"]},
                        "power": {"type": ["number","null"]}
                    },
                    "required": ["primary_metric"],
                    "additionalProperties": True
                },
                "method": {"type": "string"},
                "suggested_n": {"type": ["integer","null"]},
                "notes": {"type": "string"}
            },
            "required": ["assumptions","method"],
            "additionalProperties": False
        },
        "recommended_designs": {
            "type": "array", "minItems": 1, "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rationale": {"type": "string", "minLength": 30},
                    "steps": {"type": "array", "minItems": 2, "maxItems": 10,
                              "items": {"type": "string", "minLength": 8}}
                },
                "required": ["name","rationale","steps"],
                "additionalProperties": False
            }
        },
        "ablations": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "why": {"type": "string", "minLength": 15},
                    "how": {"type": "string", "minLength": 15}
                },
                "required": ["name","why","how"],
                "additionalProperties": False
            }
        },
        "validation": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},  # e.g., k-fold CV, holdout, temporal split, calibration
                    "details": {"type": "string", "minLength": 15}
                },
                "required": ["type","details"],
                "additionalProperties": False
            }
        },
        "risks": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "risk": {"type": "string"},
                    "mitigation": {"type": "string", "minLength": 15}
                },
                "required": ["risk","mitigation"],
                "additionalProperties": False
            }
        },
        "timeline": {
            "type": "array", "minItems": 1, "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string"},
                    "duration_days": {"type": ["integer","null"]},
                    "deliverables": {"type": "string", "minLength": 10}
                },
                "required": ["phase","deliverables"],
                "additionalProperties": False
            }
        }
    },
    "required": ["recommended_designs"],
    "additionalProperties": False
}

def build_study_design_with_llm(
    question: str,
    plan: dict,
    results: dict,
    sources: list,
    constraints: dict | None = None,
    max_tokens: int = 900
):
    """
    Return a structured study-design recommendation grounded in current results.
    """
    sys = (
        "You are a meticulous ML research designer. Propose a rigorous follow-up study "
        "based ONLY on the provided results, plan, and quoted sources. Prefer conservative, "
        "reproducible choices. Include power assumptions and a concrete sample-size suggestion "
        "when feasible. Avoid speculative claims. If the question is biomedical/clinical, DO NOT "
        "make causal or treatment claims—limit to model metrics or stated quotes. If evidence is insufficient, say so."
    )

    # Pack sources for prompt (clipped & sanitized)
    src_str = _pack_sources(sources, max_items=5)

    user_payload = {
        "question": question,
        "plan": plan,
        "results": results,
        "sources": src_str or "(no external sources)",
        "constraints": constraints or {"budget": None, "compute": None, "deadline_days": None}
    }

    client = LLMClient()
    data, meta = _retry_json(
        client,
        sys,
        "Using ONLY the JSON below, output a study-design plan that validates against the schema. "
        "Every numeric claim must be copied from the 'results' JSON or the quoted sources. Do not invent numbers."
        "\n\nINPUT:\n" + json.dumps(user_payload, ensure_ascii=False),
        JSON_SCHEMA_STUDY,
        max_tokens=max_tokens
    )
    return data, meta

def build_conclusion_with_llm(
    question: str,
    plan: dict,
    results: dict,
    sources: list,
    hypotheses: list,
    max_tokens: int = 900
):
    """
    Produce a detailed, conservative, reproducible Conclusion based on:
    - question, plan (design/primary_metric/etc.), results (metrics/p-values),
    - sources (quotes), and the hypotheses considered.
    Returns (data, meta) where data validates JSON_SCHEMA_CONCLUSION.
    """
    sys = (
        "You are a careful research analyst. Write a rigorous, detailed conclusion grounded in the provided "
        "metrics and quotes. Be conservative about causality and generalization. Explicitly reference numeric "
        "results and uncertainty. No hallucination; use only provided info. If the question is biomedical/clinical, "
        "DO NOT make causal or treatment claims; if evidence is insufficient, state that clearly."
    )

    # Pack sources (clipped & sanitized)
    src_str = _pack_sources(sources, max_items=5)

    user = {
        "question": question,
        "plan": plan,            # include design/primary_metric/seed/test_size/alpha etc.
        "results": results,      # ML metrics or t-test stats
        "hypotheses": hypotheses or [],
        "sources": src_str or "(no external sources)"
    }

    client = LLMClient()
    data, meta = client.chat_json(
        sys,
        "Using ONLY the provided JSON below, produce a structured conclusion.\n\nINPUT:\n" + json.dumps(user, ensure_ascii=False),
        JSON_SCHEMA_CONCLUSION,
        max_tokens=max_tokens
    )
    return data, meta

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Be forgiving: strip code fences; then parse full or largest top-level block."""
    if not text:
        return None
    t = text.strip()
    # Strip common ```json fences
    if t.startswith("```"):
        t = t.strip("`")
        if "\n" in t and t[:10].lower().lstrip().startswith(("json", "javascript")):
            t = t.split("\n", 1)[1]
        t = t.strip()
    # Fast path
    try:
        return json.loads(t)
    except Exception:
        pass
    # Largest top-level block
    try:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            candidate = t[start:end+1]
            return json.loads(candidate)
    except Exception:
        return None
    return None

def _pack_sources(sources: list, max_items: int = 5) -> str:
    """Compact, clip and sanitize sources for prompts to keep tokens in check."""
    buf: List[str] = []
    for i, s in enumerate((sources or [])[:max_items], start=1):
        title = (s.get("title") or (s.get("source") or {}).get("name") or "Untitled").strip()
        quote = (s.get("quote") or "").replace("\n", " ").strip()
        url = (s.get("url") or (s.get("source") or {}).get("name") or "").strip()
        if len(quote) > 420:
            quote = quote[:420].rsplit(" ", 1)[0] + "…"
        buf.append(f"[{i}] {title}\nURL: {url}\n> {quote}")
    return ("\n" + "\n".join(buf)) if buf else "(no external sources)"

def _retry_json(client, system: str, user: str, schema: Dict[str, Any], max_tokens: int):
    """One strict retry with a JSON-only clamp on validation/parse errors."""
    try:
        return client.chat_json(system, user, schema, max_tokens)
    except (ValidationError, ValueError, RuntimeError):
        stricter_user = (
            "OUTPUT STRICT JSON ONLY. NO prose. NO code fences. "
            "Return a value that VALIDATES against the schema.\n\n" + user
        )
        return client.chat_json(system, stricter_user, schema, max_tokens)

class LLMClient:
    def __init__(self):
        self.provider = (os.getenv("LLM_PROVIDER") or "openai").lower()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.timeout = 30

        if self.provider == "openai":
            OpenAI = _lazy_openai()
            base_url = os.getenv("OPENAI_BASE_URL")  # optional
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        elif self.provider == "openrouter":
            OpenAI = _lazy_openai()
            # OpenRouter is OpenAI-compatible; just swap base_url + key
            self.client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL","https://openrouter.ai/api/v1")
            )
        elif self.provider == "anthropic":
            Anthropic = _lazy_anthropic()
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise RuntimeError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def _openai_json(self, system: str, user: str, schema: Dict[str, Any], max_tokens: int = 800, use_schema: bool = True) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Use Chat Completions. If use_schema=True, request structured outputs via response_format json_schema.
        Returns (data, finish_reason).
        """
        kwargs = dict(
            model=self.openai_model if self.provider == "openai" else self.openrouter_model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        if use_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "structured_output", "schema": schema, "strict": True}}
        # Handle SDK timeout kw differences (timeout vs request_timeout)
        try:
            resp = self.client.chat.completions.create(timeout=self.timeout, **kwargs)
        except TypeError:
            resp = self.client.chat.completions.create(request_timeout=self.timeout, **kwargs)
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        data = _extract_json(text)
        if not data:
            raise ValueError("Failed to parse JSON from OpenAI response")
        validate(instance=data, schema=schema)
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        return data, finish_reason

    def _anthropic_json(self, system: str, user: str, schema: Dict[str, Any], max_tokens: int = 800) -> Dict[str, Any]:
        # Anthropic doesn't hard-enforce JSON schema; we enforce via prompt + validator.
        prompt = (
            "You MUST respond with STRICT JSON ONLY that validates against this schema.\n"
            "Do not include prose before/after the JSON.\n"
            f"SCHEMA:\n{json.dumps(schema)}\n\n"
            f"USER:\n{user}"
        )
        resp = self.client.messages.create(
            model=self.anthropic_model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role":"user","content": prompt}],
            timeout=self.timeout
        )
        text_blocks = [b.text for b in (resp.content or []) if getattr(b, "type", "") == "text"]
        text = "".join(text_blocks).strip()
        data = _extract_json(text)
        if not data:
            raise ValueError("Failed to parse JSON from Anthropic response")
        validate(instance=data, schema=schema)
        return data

    def chat_json(self, system: str, user: str, schema: Dict[str, Any], max_tokens: int = 800) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns (data, meta). 'data' is schema-validated JSON; 'meta' has provider/model/timing.
        """
        t0 = time.time()
        finish_reason = None
        if self.provider == "openai":
            data, finish_reason = self._openai_json(system, user, schema, max_tokens=max_tokens, use_schema=True)
            model = self.openai_model
        elif self.provider == "openrouter":
            # Try schema-enforced first; if model/provider rejects response_format, fall back
            try:
                data, finish_reason = self._openai_json(system, user, schema, max_tokens=max_tokens, use_schema=True)
            except Exception:
                prompt = (
                    "You MUST respond with STRICT JSON ONLY that validates against this schema.\n"
                    "No prose, no code fences, no comments.\n"
                   f"SCHEMA:\n{json.dumps(schema)}\n\nUSER:\n{user}"
                )
                data, finish_reason = self._openai_json(system, prompt, schema, max_tokens=max_tokens, use_schema=False)
            model = self.openrouter_model
        elif self.provider == "anthropic":
            data = self._anthropic_json(system, user, schema, max_tokens=max_tokens)
            model = self.anthropic_model
        else:
            raise RuntimeError("Bad provider")
        dt = int((time.time() - t0)*1000)
        meta = {"provider": self.provider, "model": model, "latency_ms": dt}
        if finish_reason is not None:
            meta["finish_reason"] = finish_reason
        return data, meta

# ---------- High-level helpers ----------
def build_hypotheses_with_llm(question: str, dataset_meta: Dict[str, Any], sources: List[Dict[str,Any]], max_items: int = 3):
    sys = "You are a careful research assistant. Propose falsifiable, non-trivial hypotheses aligned with the question and sources."
    src_str = _pack_sources(sources, max_items=5)
    ds = dataset_meta or {}
    user = (
        f"QUESTION: {question or 'N/A'}\n"
        f"DATASET: name={ds.get('name')}, target={ds.get('target_name')}\n"
        f"SOURCES:{src_str or ' (none)'}\n"
        f"Write {max_items} hypotheses tailored to this context."
    )
    client = LLMClient()
    data, meta = _retry_json(client, sys, user, JSON_SCHEMA_HYPOTHESES, max_tokens=600)
    data["hypotheses"] = data["hypotheses"][:max_items]
    return data, meta

def summarize_sources_with_llm(sources: List[Dict[str,Any]], max_items: int = 3):
    sys = "You write concise, faithful TL;DRs. No fabrication. Use only the provided quotes."
    items = []
    for s in sources[:max_items]:
        title = s.get("title") or (s.get("source") or {}).get("name") or "Untitled"
        items.append({"title": title, "url": s.get("url") or (s.get("source") or {}).get("name"), "quote": s.get("quote","")})
    user = "Summarize each item into a one-sentence TL;DR using only its quote.\n" + json.dumps(items, ensure_ascii=False)
    client = LLMClient()
    data, meta = _retry_json(client, sys, user, JSON_SCHEMA_SUMMARIES, max_tokens=600)
    return data, meta
