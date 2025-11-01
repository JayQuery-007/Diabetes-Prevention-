import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from dotenv import load_dotenv

from utils.model import load_model_artifacts, predict_risk
from utils.gemini import get_gemini_recommendations, get_gemini_profile_summary

# Import visualization generator from training script
from train_model import generate_model_visualizations

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

model_bundle = load_model_artifacts()


@app.route("/")
def home():
	return render_template("index.html")


@app.route("/assessment")
def assessment():
	return render_template("assessment.html")


@app.route("/predict", methods=["POST"]) 
def predict():
	if request.is_json:
		data = request.get_json(force=True)
	else:
		data = request.form.to_dict()

	prediction = predict_risk(model_bundle, data)

	session["last_result"] = {
		"risk_score": prediction["risk_score"],
		"risk_category": prediction["risk_category"],
		"top_factors": prediction.get("top_factors", []),
		"inputs": prediction["inputs"],
	}
	# clear any old ai outputs (not used now, but keep tidy)
	session.pop("last_recs", None)
	session.pop("last_summary", None)

	return redirect(url_for("result"))


@app.route("/result")
def result():
	last_result = session.get("last_result")
	if not last_result:
		return redirect(url_for("assessment"))

	metrics = (model_bundle.meta or {}).get("metrics") if model_bundle else None

	# Compute AI outputs freshly on each render to avoid stale/static content
	recs = get_gemini_recommendations(
		last_result["inputs"],
		last_result["risk_score"],
		last_result.get("top_factors", []),
		last_result.get("risk_category", ""),
	) or []
	summary = get_gemini_profile_summary(
		last_result["inputs"],
		last_result["risk_score"],
		last_result.get("top_factors", []),
		last_result.get("risk_category", ""),
	)
	return render_template("result.html", result=last_result, recommendations=recs, metrics=metrics, profile_summary=summary, viz=None)


@app.route("/metrics", methods=["POST"]) 
def metrics_route():
	last_result = session.get("last_result")
	if not last_result:
		return redirect(url_for("assessment"))

	dataset_path = request.form.get("dataset_path") or os.path.join("models", "demo.csv")
	try:
		viz = generate_model_visualizations(dataset_path, outdir=os.path.join("static", "plots"))
		# Reload bundle meta so updated best metrics appear next render
		global model_bundle
		model_bundle = load_model_artifacts()
		metrics = (model_bundle.meta or {}).get("metrics") if model_bundle else None

		# Compute AI outputs freshly here as well so they remain dynamic
		recs = get_gemini_recommendations(
			last_result["inputs"],
			last_result["risk_score"],
			last_result.get("top_factors", []),
			last_result.get("risk_category", ""),
		) or []
		summary = get_gemini_profile_summary(
			last_result["inputs"],
			last_result["risk_score"],
			last_result.get("top_factors", []),
			last_result.get("risk_category", ""),
		)

		return render_template("result.html", result=last_result, recommendations=recs, metrics=metrics, profile_summary=summary, viz=viz)
	except Exception as e:
		return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
	port = int(os.getenv("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=True)
