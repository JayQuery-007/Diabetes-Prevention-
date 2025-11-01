import os
import argparse
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')  # non-GUI backend for server environments
import matplotlib.pyplot as plt

# Try to import XGBoost; if unavailable, we will skip it gracefully
try:
	from xgboost import XGBClassifier
	XGB_AVAILABLE = True
except Exception:
	XGB_AVAILABLE = False

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
TARGET = "Diabetes"


def load_data(input_csv: str) -> pd.DataFrame:
	df = pd.read_csv(input_csv)
	missing = [f for f in EXPECTED_FEATURES + [TARGET] if f not in df.columns]
	if missing:
		raise ValueError(f"Missing columns in data: {missing}")
	return df


def build_pipeline(model):
	numeric = ["Age", "BMI"]
	categorical = [
		"Gender",
		"Diet_Type",
		"Physical_Activity",
		"Stress_Level",
		"Sleep_Quality",
		"Family_History",
		"Hypertension",
		"Cholesterol_Level",
	]

	numeric_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
	])
	categorical_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", OneHotEncoder(handle_unknown="ignore")),
	])
	preprocess = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric),
			("cat", categorical_transformer, categorical),
		]
	)
	pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
	return pipe


def evaluate(pipe, x_test, y_test) -> Dict[str, Any]:
	proba = pipe.predict_proba(x_test)[:, 1]
	pred = (proba >= 0.5).astype(int)
	auc = roc_auc_score(y_test, proba)
	acc = accuracy_score(y_test, pred)
	prec = precision_score(y_test, pred, zero_division=0)
	rec = recall_score(y_test, pred, zero_division=0)
	f1 = f1_score(y_test, pred, zero_division=0)
	n00, n01, n10, n11 = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
	return {
		"roc_auc": float(auc),
		"accuracy": float(acc),
		"precision": float(prec),
		"recall": float(rec),
		"f1": float(f1),
		"confusion_matrix": {"tn": int(n00), "fp": int(n01), "fn": int(n10), "tp": int(n11)},
	}


def _get_feature_names_from_preprocessor(preprocess: ColumnTransformer) -> List[str]:
	# Build output feature names from ColumnTransformer (numeric + one-hot categorical)
	numeric_features = preprocess.transformers_[0][2]
	categorical_features = preprocess.transformers_[1][2]
	onehot = preprocess.transformers_[1][1].named_steps['onehot']
	categorical_expanded = list(onehot.get_feature_names_out(categorical_features))
	return list(numeric_features) + categorical_expanded


def _collect_feature_importance(pipeline: Pipeline) -> Dict[str, float]:
	# Map model coefficients/importances back to feature names
	pre = pipeline.named_steps['preprocess']
	feature_names = _get_feature_names_from_preprocessor(pre)
	model = pipeline.named_steps['model']
	importances: Dict[str, float] = {}
	if hasattr(model, 'coef_'):
		coef = np.abs(model.coef_[0])
		for name, val in zip(feature_names, coef):
			importances[name] = float(val)
	elif hasattr(model, 'feature_importances_'):
		for name, val in zip(feature_names, model.feature_importances_):
			importances[name] = float(val)
	else:
		# Fallback: zero importances
		for name in feature_names:
			importances[name] = 0.0
	return importances


def generate_model_visualizations(data_csv: str, outdir: str = "static/plots") -> Dict[str, Any]:
	"""
	Train Logistic Regression, Random Forest, and XGBoost (if available), compute metrics,
	generate correlation heatmap, feature importance comparison (top 10), and model metrics chart.
	Saves PNGs to static/plots and returns { metrics_by_model, plot_paths }.
	"""
	os.makedirs(outdir, exist_ok=True)
	df = load_data(data_csv)

	x = df[EXPECTED_FEATURES]
	y = df[TARGET]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

	candidates: Dict[str, Any] = {
		"logreg": LogisticRegression(max_iter=1000),
		"rf": RandomForestClassifier(n_estimators=400, random_state=42),
	}
	if XGB_AVAILABLE:
		candidates["xgb"] = XGBClassifier(
			max_depth=4,
			n_estimators=400,
			learning_rate=0.05,
			subsample=0.9,
			colsample_bytree=0.9,
			eval_metric='logloss',
			random_state=42,
		)

	trained: Dict[str, Pipeline] = {}
	metrics: Dict[str, Dict[str, Any]] = {}
	importances_by_model: Dict[str, Dict[str, float]] = {}

	best_name, best_auc = None, -1.0
	for name, mdl in candidates.items():
		pipe = build_pipeline(mdl)
		pipe.fit(x_train, y_train)
		trained[name] = pipe
		m = evaluate(pipe, x_test, y_test)
		metrics[name] = m
		if m['roc_auc'] > best_auc:
			best_auc, best_name = m['roc_auc'], name
		# feature importances
		try:
			importances_by_model[name] = _collect_feature_importance(pipe)
		except Exception:
			importances_by_model[name] = {}

	# Save the best model as the primary artifact
	os.makedirs('models', exist_ok=True)
	joblib.dump(trained[best_name], os.path.join('models', 'diabetes_model.joblib'))
	joblib.dump({}, os.path.join('models', 'label_encoders.joblib'))
	joblib.dump({
		"features": EXPECTED_FEATURES,
		"target": TARGET,
		"model": best_name,
		"metrics": metrics.get(best_name),
	}, os.path.join('models', 'meta.joblib'))

	# ========== FIXED CORRELATION HEATMAP - SHOWS ALL FEATURES (Matplotlib only) ==========
	# Encode categorical variables to numeric for correlation calculation
	df_encoded = df.copy()
	label_encoders = {}
	
	categorical_cols = [col for col in EXPECTED_FEATURES if not pd.api.types.is_numeric_dtype(df[col])]
	
	for col in categorical_cols:
		le = LabelEncoder()
		df_encoded[col] = le.fit_transform(df[col].astype(str))
		label_encoders[col] = le
	
	# Calculate correlation matrix for ALL features + target
	corr = df_encoded[EXPECTED_FEATURES + [TARGET]].corr()
	
	# Create heatmap using matplotlib only
	fig, ax = plt.subplots(figsize=(12, 10))
	
	# Create color map (red-white-blue)
	cmap = plt.cm.RdBu_r
	im = ax.imshow(corr, cmap=cmap, aspect='auto', vmin=-1, vmax=1, interpolation='nearest')
	
	# Add colorbar
	cbar = plt.colorbar(im, ax=ax, shrink=0.8)
	cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=11)
	
	# Set ticks and labels
	feature_names = EXPECTED_FEATURES + [TARGET]
	ax.set_xticks(np.arange(len(feature_names)))
	ax.set_yticks(np.arange(len(feature_names)))
	ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
	ax.set_yticklabels(feature_names, fontsize=10)
	
	# Add correlation values as text annotations
	for i in range(len(feature_names)):
		for j in range(len(feature_names)):
			text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
						   ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white",
						   fontsize=8)
	
	# Add gridlines
	ax.set_xticks(np.arange(len(feature_names)) - 0.5, minor=True)
	ax.set_yticks(np.arange(len(feature_names)) - 0.5, minor=True)
	ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
	ax.tick_params(which="minor", size=0)
	
	plt.title('Feature Correlation Heatmap (All Features)', fontsize=16, fontweight='bold', pad=20)
	plt.xlabel('Features', fontsize=12, labelpad=10)
	plt.ylabel('Features', fontsize=12, labelpad=10)
	plt.tight_layout()
	corr_path = os.path.join(outdir, 'correlation_heatmap.png')
	plt.savefig(corr_path, dpi=200, bbox_inches='tight')
	plt.close()
	# =====================================================================

	# Feature importance comparison (top 10 by average importance across models)
	all_features = set()
	for imp in importances_by_model.values():
		all_features.update(imp.keys())
	avg_scores = []
	for f in all_features:
		vals = [importances_by_model[m].get(f, 0.0) for m in importances_by_model]
		avg_scores.append((f, float(np.mean(vals))))
	top10 = [f for f, _ in sorted(avg_scores, key=lambda x: x[1], reverse=True)[:10]]

	indices = np.arange(len(top10))
	width = 0.25
	plt.figure(figsize=(8,5))
	colors = {'logreg': '#0d6efd', 'rf': '#20c997', 'xgb': '#ffc107'}
	for i, (name, _) in enumerate(candidates.items()):
		vals = [importances_by_model.get(name, {}).get(f, 0.0) for f in top10]
		plt.bar(indices + (i-1)*width, vals, width=width, label=name.upper(), color=colors.get(name, '#6c757d'))
	plt.xticks(indices, top10, rotation=45, ha='right')
	plt.ylabel('Importance (normalized)')
	plt.title('Top 10 Feature Importance Across Models')
	plt.legend()
	plt.tight_layout()
	feat_path = os.path.join(outdir, 'feature_importance.png')
	plt.savefig(feat_path, dpi=160)
	plt.close()

	# Model metrics comparison chart
	metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
	indices = np.arange(len(metric_names))
	width = 0.25
	plt.figure(figsize=(8,4.5))
	for i, name in enumerate(metrics.keys()):
		vals = [metrics[name][m] for m in metric_names]
		plt.bar(indices + (i-1)*width, vals, width=width, label=name.upper(), color=colors.get(name, '#6c757d'))
	plt.xticks(indices, [m.upper() for m in metric_names], rotation=0)
	plt.ylim(0, 1.0)
	plt.title('Model Performance Comparison')
	plt.legend()
	plt.tight_layout()
	metrics_path = os.path.join(outdir, 'model_metrics.png')
	plt.savefig(metrics_path, dpi=160)
	plt.close()

	return {
		"best_model": best_name,
		"metrics_by_model": metrics,
		"plots": {
			"correlation_heatmap": '/' + corr_path.replace('\\', '/'),
			"feature_importance": '/' + feat_path.replace('\\', '/'),
			"model_metrics": '/' + metrics_path.replace('\\', '/'),
		},
	}


def main():
	parser = argparse.ArgumentParser(description="Train diabetes risk model")
	parser.add_argument("--data", required=True, help="Path to merged training CSV with expected columns and target")
	parser.add_argument("--outdir", default="models", help="Directory to save model artifacts")
	args = parser.parse_args()

	result = generate_model_visualizations(args.data, outdir="static/plots")
	# Save best model meta is already handled inside generate_model_visualizations
	print(f"Best model: {result['best_model']}")
	print("Saved plots:", result['plots'])


if __name__ == "__main__":
	main()