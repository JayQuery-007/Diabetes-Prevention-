import os
from typing import Any, Dict, List, Optional
import re
import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyD49OuUU0-XKGlxV4OLVc-278Q44toQxjI"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT_V1 = "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
GEMINI_ENDPOINT_V1B = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT_TEMPLATE = (
	"You are a health coach. Based on the user's profile, risk score, category, and contributing factors, "
	"suggest 3–5 specific, safe, and practical improvements over the next 4–6 weeks to reduce diabetes risk. "
	"Use concise bullet points (no numbering).\n\n"
	"Risk Score: {score}%\n"
	"Risk Category: {category}\n"
	"Contributing Factors: {factors}\n\n"
	"User Profile (JSON):\n{profile}"
)

SUMMARY_PROMPT = (
	"Summarize the user's lifestyle profile and risk in 3–5 sentences. "
	"Mention BMI, activity, stress, sleep, diet, and any risk elevating factors. "
	"Be concise and neutral.\n\n"
	"Risk Score: {score}%\n"
	"Risk Category: {category}\n"
	"Contributing Factors: {factors}\n\n"
	"User Profile (JSON):\n{profile}"
)


def _tailored_local_fallback(profile: Dict[str, Any], score: float, factors: Optional[List[str]]) -> List[str]:
	factors = factors or []
	recs: List[str] = []
	recs.append("Aim for 7–8 hours nightly; consistent sleep-wake times and a dark, cool room.")
	recs.append("Replace sugary drinks with water; choose whole grains and add 1 palm-sized protein each meal.")
	lower_factors = [f.lower() for f in factors]
	if any("bmi" in f for f in lower_factors):
		recs.append("Create a 300–500 kcal daily deficit; walk 30–45 min most days and add 2 strength sessions/week.")
	if any("stress" in f for f in lower_factors):
		recs.append("Do 10 minutes of breathwork or guided meditation daily; schedule 2 short breaks during work/study.")
	if any("sleep" in f for f in lower_factors):
		recs.append("Establish a 60‑minute wind‑down routine; avoid screens and heavy meals close to bedtime.")
	if any("activity" in f for f in lower_factors):
		recs.append("Accumulate 8,000–10,000 steps/day; take a 10‑minute walk after main meals.")
	if any("cholesterol" in f for f in lower_factors):
		recs.append("Use healthy fats (olive/mustard oil); add nuts/seeds 5x/week; limit fried foods.")
	if any("family" in f for f in lower_factors):
		recs.append("Check fasting glucose in 3–6 months; maintain routine annual screening given family history.")
	return recs[:5]


def _parse_candidates_text(data: Dict[str, Any]) -> str:
	candidates = data.get("candidates") or []
	if not candidates:
		return ""
	content = candidates[0].get("content") or {}
	parts = content.get("parts") or []
	text = "\n".join([p.get("text", "") for p in parts if isinstance(p, dict)])
	return text.strip()


def _split_into_items(text: str) -> List[str]:
	lines = [ln.strip("- •\t ") for ln in text.splitlines() if ln.strip()]
	if lines:
		return lines
	sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
	return sentences


def _try_gemini(prompt: str) -> List[str]:
	payloads = [
		{"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.6, "topP": 0.9, "maxOutputTokens": 512}},
		{"contents": [{"parts": [{"text": prompt}]}]},
	]
	strategies = [
		{"url": GEMINI_ENDPOINT_V1.format(model=GEMINI_MODEL), "use_header": True},
		{"url": GEMINI_ENDPOINT_V1.format(model=GEMINI_MODEL), "use_header": False},
		{"url": GEMINI_ENDPOINT_V1B.format(model=GEMINI_MODEL), "use_header": True},
		{"url": GEMINI_ENDPOINT_V1B.format(model=GEMINI_MODEL), "use_header": False},
	]
	for strat in strategies:
		for payload in payloads:
			try:
				headers = {"Content-Type": "application/json"}
				params = None
				if strat["use_header"]:
					headers["x-goog-api-key"] = GEMINI_API_KEY
				else:
					params = {"key": GEMINI_API_KEY}
				resp = requests.post(strat["url"], json=payload, headers=headers, params=params, timeout=25)
				if resp.status_code >= 400:
					print("Gemini HTTP error:", resp.status_code, resp.text)
					continue
				data = resp.json()
				text = _parse_candidates_text(data)
				if text:
					items = _split_into_items(text)
					if items:
						return items[:5]
				else:
					# safety block or empty; log promptFeedback if present
					pf = (data.get("promptFeedback") or {}).get("blockReason", "")
					if pf:
						print("Gemini block:", pf)
			except Exception as e:
				print("Gemini exception:", str(e))
	return []


def get_gemini_recommendations(profile: Dict[str, Any], score: float, factors: Optional[List[str]] = None, category: str = "") -> List[str]:
	if not GEMINI_API_KEY:
		return _tailored_local_fallback(profile, score, factors)
	prompt = PROMPT_TEMPLATE.format(profile=str(profile), score=round(float(score), 1), category=category or "", factors=", ".join(factors or []) or "(none)")
	items = _try_gemini(prompt)
	if items:
		return items
	return _tailored_local_fallback(profile, score, factors)


def get_gemini_profile_summary(profile: Dict[str, Any], score: float, factors: Optional[List[str]] = None, category: str = "") -> str:
	if not GEMINI_API_KEY:
		return "Profile indicates areas to optimize in activity, stress, sleep, and diet."
	prompt = SUMMARY_PROMPT.format(profile=str(profile), score=round(float(score), 1), category=category or "", factors=", ".join(factors or []) or "(none)")
	items = _try_gemini(prompt)
	if items:
		return " ".join(items[:3])
	return "Profile indicates areas to optimize in activity, stress, sleep, and diet."
