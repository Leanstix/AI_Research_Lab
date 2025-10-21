# app/core/llm.py
# Stubs that can call OpenAI/Anthropic if ENABLE_LLM=true and keys are set.
import os, json, time
from typing import Any, Dict, Tuple, List
from jsonschema import validate

def _lazy_openai():
    from openai import OpenAI
    return OpenAI

JSON_SCHEMA_HYPOTHESES = {
    "type": "object",
    "properties": { "hypotheses": { "type": "array", "items": {"type":"string"}, "minItems": 1, "maxItems": 6 } },
    "required": ["hypotheses"], "additionalProperties": False
}

JSON_SCHEMA_SUMMARIES = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "items": { "type": "object",
                "properties": {"title":{"type":"string"}, "url":{"type":"string"}, "tldr":{"type":"string"}},
                "required": ["title","url","tldr"], "additionalProperties": False }
        }
    },
    "required": ["summaries"], "additionalProperties": False
}

JSON_SCHEMA_CONCLUSION = {
    "type": "object",
    "properties": {
        "conclusion": {"type": "string"},
        "strength_of_evidence": {"type": "string"},
        "limitations": {"type": "array", "items": {"type":"string"}},
        "next_steps": {"type": "array", "items": {"type":"string"}}
    },
    "required": ["conclusion","strength_of_evidence","limitations","next_steps"],
    "additionalProperties": False
}

JSON_SCHEMA_STUDY = {
    "type": "object",
    "properties": {
        "recommended_designs": {"type": "array", "items": {"type":"object"}, "minItems":1}
    },
    "required": ["recommended_designs"], "additionalProperties": True
}

class LLMClient:
    def __init__(self):
        self.provider = "openai"
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = 30
        OpenAI = _lazy_openai()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat_json(self, system: str, user: str, schema: Dict[str, Any], max_tokens: int = 500):
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            max_tokens=max_tokens, timeout=self.timeout
        )
        text = (resp.choices[0].message.content or "").strip()
        data = json.loads(text)
        validate(instance=data, schema=schema)
        return data, {"provider":"openai","model":self.model}

def build_hypotheses_with_llm(question: str, dataset_meta: Dict[str, Any], sources: List[Dict[str,Any]], max_items: int = 3):
    sys = "You are a careful research assistant."
    items = []
    for s in sources[:max_items]:
        title = s.get("title") or (s.get("source") or {}).get("name") or "Untitled"
        items.append({"title": title, "url": s.get("url")})
    user = json.dumps({"question": question, "dataset": dataset_meta, "sources": items, "count": max_items})
    client = LLMClient()
    data, meta = client.chat_json(sys, user, JSON_SCHEMA_HYPOTHESES, max_tokens=400)
    data["hypotheses"] = data.get("hypotheses", [])[:max_items]
    return data, meta

def summarize_sources_with_llm(sources: List[Dict[str,Any]], max_items: int = 3):
    sys = "Summarize each item into a one-sentence TL;DR using only its quote."
    items = []
    for s in sources[:max_items]:
        items.append({"title": s.get("title") or "Untitled", "url": s.get("url"), "quote": s.get("quote","")})
    user = json.dumps(items, ensure_ascii=False)
    client = LLMClient()
    return client.chat_json(sys, user, JSON_SCHEMA_SUMMARIES, max_tokens=400)

def build_conclusion_with_llm(question: str, plan: dict, results: dict, sources: list, hypotheses: list, max_tokens: int = 600):
    sys = "Write a cautious, grounded conclusion using only provided metrics and quotes."
    payload = {"question":question, "plan":plan, "results":results, "sources":sources, "hypotheses":hypotheses}
    client = LLMClient()
    return client.chat_json(sys, json.dumps(payload), JSON_SCHEMA_CONCLUSION, max_tokens=max_tokens)

def build_study_design_with_llm(question: str, plan: dict, results: dict, sources: list, constraints: dict | None = None, max_tokens: int = 600):
    sys = "Propose a reproducible follow-up study with conservative choices."
    payload = {"question":question, "plan":plan, "results":results, "sources":sources, "constraints":constraints or {}}
    client = LLMClient()
    return client.chat_json(sys, json.dumps(payload), JSON_SCHEMA_STUDY, max_tokens=max_tokens)