import os
from typing import Any, Dict, Tuple
import math

try:
	import joblib
except Exception:  # pragma: no cover
	joblib = None


EXPECTED_FEATURES = [
	"Age",
	"Gender",
	"BMI",
	"Diet_Type",
	"Physical_Activity",
	"Stress_Level",
	"Sleep_Quality",
	"Family_History",
	"Hypertension",
	"Cholesterol_Level",
]


class ModelBundle:
	def __init__(self, model=None, encoders=None, meta=None):
		self.model = model
		self.encoders = encoders or {}
		self.meta = meta or {}


def load_model_artifacts() -> ModelBundle:
	models_dir = os.path.join(os.getcwd(), "models")
	model_path = os.path.join(models_dir, "diabetes_model.joblib")
	encoders_path = os.path.join(models_dir, "label_encoders.joblib")
	meta_path = os.path.join(models_dir, "meta.joblib")

	model = encoders = meta = None
	if joblib and os.path.exists(model_path):
		try:
			model = joblib.load(model_path)
			if os.path.exists(encoders_path):
				encoders = joblib.load(encoders_path)
			if os.path.exists(meta_path):
				meta = joblib.load(meta_path)
		except Exception:
			model = None

	return ModelBundle(model=model, encoders=encoders, meta=meta)


def _parse_float(value: Any, default: float = 0.0) -> float:
	try:
		return float(value)
	except Exception:
		return default


def _bmi_from_height_weight(height_cm: Any, weight_kg: Any) -> float:
	height_m = _parse_float(height_cm) / 100.0
	weight = _parse_float(weight_kg)
	if height_m <= 0:
		return 0.0
	return weight / (height_m ** 2)


def _normalize_inputs(raw: Dict[str, Any]) -> Dict[str, Any]:
	# Allow BMI auto-calc from height and weight if provided
	inputs = dict(raw)
	if (not inputs.get("BMI")) and inputs.get("Height") and inputs.get("Weight"):
		inputs["BMI"] = round(_bmi_from_height_weight(inputs.get("Height"), inputs.get("Weight")), 1)

	# Provide sane defaults
	defaults = {
		"Age": 22,
		"Gender": "Male",
		"BMI": 24.0,
		"Diet_Type": "Balanced",
		"Physical_Activity": "Moderate",
		"Stress_Level": "Medium",
		"Sleep_Quality": "Average",
		"Family_History": "No",
		"Hypertension": "No",
		"Cholesterol_Level": "Normal",
	}
	for k, v in defaults.items():
		inputs.setdefault(k, v)
		# Strip strings
		if isinstance(inputs[k], str):
			inputs[k] = inputs[k].strip()
	return inputs


def _heuristic_score(inputs: Dict[str, Any]) -> Tuple[float, str, list]:
	# Baseline risk
	risk = 0.20  # 20%

	bmi = _parse_float(inputs.get("BMI"), 24.0)
	age = _parse_float(inputs.get("Age"), 22.0)

	# BMI contribution
	if bmi < 18.5:
		risk += 0.02
	elif 25 <= bmi < 30:
		risk += 0.08
	elif 30 <= bmi < 35:
		risk += 0.18
	elif 35 <= bmi < 40:
		risk += 0.28
	elif bmi >= 40:
		risk += 0.45

	# Age: slightly increases with age in young adults
	if age >= 30:
		risk += 0.05

	# Stress level
	stress = str(inputs.get("Stress_Level", "Medium")).lower()
	if stress in ["high", "7", "8", "9", "10"]:
		risk += 0.15
	elif stress in ["low", "1", "2", "3"]:
		risk -= 0.03

	# Sleep quality
	sleep = str(inputs.get("Sleep_Quality", "Average")).lower()
	if sleep in ["poor", "bad"]:
		risk += 0.12
	elif sleep in ["good", "excellent"]:
		risk -= 0.05

	# Diet
	diet = str(inputs.get("Diet_Type", "Balanced")).lower()
	if "fast" in diet or "irregular" in diet:
		risk += 0.10

	# Physical activity
	activity = str(inputs.get("Physical_Activity", "Moderate")).lower()
	if activity in ["sedentary", "low", "none"]:
		risk += 0.10
	elif activity in ["high", "active", "vigorous"]:
		risk -= 0.05

	# Family history
	if str(inputs.get("Family_History", "No")).lower() in ["yes", "1", "true"]:
		risk += 0.12

	# Hypertension
	if str(inputs.get("Hypertension", "No")).lower() in ["yes", "1", "true"]:
		risk += 0.10

	# Cholesterol
	chol = str(inputs.get("Cholesterol_Level", "Normal")).lower()
	if chol in ["high", "very high", "very_high", "veryhigh"]:
		risk += 0.15

	# Clamp
	risk = max(0.05, min(0.95, risk))

	# Category
	if risk <= 0.35:
		category = "Low"
	elif risk <= 0.69:
		category = "Moderate"
	else:
		category = "High"

	# Top factors (simple ranking)
	factors = []
	if bmi >= 30:
		factors.append("High BMI")
	if stress in ["high", "7", "8", "9", "10"]:
		factors.append("High stress")
	if sleep in ["poor", "bad"]:
		factors.append("Poor sleep")
	if activity in ["sedentary", "low", "none"]:
		factors.append("Low activity")
	if str(inputs.get("Family_History", "No")).lower() in ["yes", "1", "true"]:
		factors.append("Family history")
	if chol in ["high", "very high", "very_high", "veryhigh"]:
		factors.append("High cholesterol")

	return risk, category, factors[:5]


def predict_risk(bundle: ModelBundle, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
	inputs = _normalize_inputs(raw_inputs)

	# Always compute heuristic factors for explanation
	_, _, heuristic_factors = _heuristic_score(inputs)

	if bundle and bundle.model is not None and joblib is not None:
		# Use the real model if available
		try:
			import pandas as pd
			row = {f: inputs.get(f) for f in EXPECTED_FEATURES}
			X = pd.DataFrame([row])
			proba = None
			if hasattr(bundle.model, "predict_proba"):
				proba = float(bundle.model.predict_proba(X)[0][1])
			else:
				pred = float(bundle.model.predict(X)[0])
				proba = max(0.0, min(1.0, pred))

			# Simple category mapping
			if proba <= 0.35:
				category = "Low"
			elif proba <= 0.69:
				category = "Moderate"
			else:
				category = "High"

			return {
				"risk_score": round(proba * 100, 1),
				"risk_category": category,
				"top_factors": heuristic_factors,  # explain with heuristic factors for clarity
				"inputs": inputs,
			}
		except Exception:
			pass

	# Fallback heuristic aligned to spec
	risk, category, factors = _heuristic_score(inputs)
	return {
		"risk_score": round(risk * 100, 1),
		"risk_category": category,
		"top_factors": factors,
		"inputs": inputs,
	}
