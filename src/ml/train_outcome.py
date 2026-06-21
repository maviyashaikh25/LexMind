import pandas as pd, numpy as np
import xgboost as xgb, shap, json
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

COURT_RANK = {
    "Supreme Court": 4, "High Court": 3,
    "District Court": 2, "Tribunal": 1
}
CHARGE_LABELS = ["302", "307", "420", "498A", "376", "304B", "other"]

def get_court_level(court_name: str) -> int:
    if not isinstance(court_name, str):
        return 1
    c_lower = court_name.lower()
    if "supreme court" in c_lower:
        return 4
    elif "high court" in c_lower:
        return 3
    elif "district court" in c_lower or "sessions court" in c_lower or "sessions judge" in c_lower:
        return 2
    elif "tribunal" in c_lower:
        return 1
    return 1

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["court_level"]   = df["court"].apply(get_court_level)
    df["text_len"]      = df["text_length"].fillna(0)
    
    text_source = df["cleaned_text"].fillna("").str[:1000] + " " + df["title"].fillna("")
    df["has_bail"]      = text_source.str.contains("bail", case=False).astype(int)
    df["has_appeal"]    = text_source.str.contains("appeal", case=False).astype(int)
    df["is_criminal"]   = text_source.str.contains("criminal|IPC|CrPC", case=False).astype(int)

    for ch in CHARGE_LABELS:
        # Use word boundaries to search for exact section numbers in clean text or title
        df[f"charge_{ch}"] = (df["cleaned_text"].fillna("").str.contains(rf"\b{ch}\b", case=False, regex=True) | 
                              df["title"].fillna("").str.contains(rf"\b{ch}\b", case=False, regex=True)).astype(int)

    FEATURE_COLS = [
        "court_level", "text_len",
        "has_bail", "has_appeal", "is_criminal"
    ] + [f"charge_{ch}" for ch in CHARGE_LABELS]

    return df[FEATURE_COLS], FEATURE_COLS

def build_features_single(entities_dict: dict, court: str, feat_names: list, raw_text: str = "") -> dict:
    court_level = get_court_level(court)
    
    # Use first 1000 characters of raw text as a proxy for the case title/header, or fallback to VERDICT/CHARGE entities
    title_text = raw_text[:1000] if raw_text else " ".join(entities_dict.get("VERDICT", []) + entities_dict.get("CHARGE", []))
    sections_text = " ".join(entities_dict.get("SECTION", []))
    
    features = {
        "court_level": court_level,
        "text_len": len(raw_text) if raw_text else 1000,
        "has_bail": int("bail" in title_text.lower()),
        "has_appeal": int("appeal" in title_text.lower()),
        "is_criminal": int(any(x in title_text.lower() for x in ["criminal", "ipc", "crpc"]))
    }
    
    for ch in CHARGE_LABELS:
        features[f"charge_{ch}"] = int(ch in sections_text or ch in raw_text)
        
    return features

LABELS = {0: "Appeal Dismissed", 1: "Appeal Allowed", 2: "Partly Allowed"}

def predict_outcome(features_dict: dict, xgb_model=None, feat_names=None) -> dict:
    if xgb_model is None:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("models/outcome_xgb.json")
    if feat_names is None:
        feat_names = json.load(open("models/feature_cols.json"))
        
    x_row = np.array([[features_dict.get(f, 0) for f in feat_names]])
    proba = xgb_model.predict_proba(x_row)[0]
    pred  = int(np.argmax(proba))
    
    sv = shap.TreeExplainer(xgb_model).shap_values(x_row)
    if isinstance(sv, list):
        sv_pred = sv[pred][0]
    elif len(sv.shape) == 3:
        sv_pred = sv[0, :, pred]
    else:
        sv_pred = sv[0]
        
    top_feats = sorted(
        zip(feat_names, sv_pred), key=lambda x: abs(x[1]), reverse=True
    )[:5]
    
    return {
        "label": LABELS[pred],
        "confidence": round(float(proba[pred]), 3),
        "probabilities": {LABELS[i]: round(float(p), 3) for i, p in enumerate(proba)},
        "top_factors": [{"feature": f, "shap": round(float(s), 4)} for f, s in top_feats],
    }

if __name__ == "__main__":
    import os
    data_path = "data/cases_metadata.csv"
    if not os.path.exists(data_path):
         print(f"Error: {data_path} not found. Please run preprocess.py first.")
         exit(1)
         
    df = pd.read_csv(data_path)
    df = df[df["outcome"] >= 0].copy()

    X, feat_cols = build_features(df)
    y = df["outcome"].values

    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )

    try:
        min_class_count = df["outcome"].value_counts().min()
        if min_class_count >= 5:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
            print(f"CV F1-macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        else:
            print(f"Warning: Too few samples per class (minimum class count: {min_class_count}) for Stratified 5-Fold CV. Skipping CV scoring.")
    except Exception as e:
        print(f"Could not calculate CV score: {e}")

    # Train/Test evaluation split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print(f"Evaluating outcome classifier using test split: {len(X_train)} train, {len(X_test)} test")
    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
    eval_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )
    eval_model.fit(X_train, y_train, sample_weight=sample_weights_train)
    y_pred = eval_model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    # Use actual present labels to avoid ValueError
    present_labels = sorted(list(set(y_test)))
    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=[LABELS[i] for i in present_labels],
        output_dict=True,
        zero_division=0
    )

    print("\n--- Outcome Classifier Test Evaluation Metrics ---")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Macro F1-Score:   {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=[LABELS[i] for i in present_labels], zero_division=0))
    print("--------------------------------------------------\n")

    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "classification_report": report
    }

    # Train final model on full dataset
    print("Fitting final model on full dataset...")
    model.fit(X, y, sample_weight=sample_weights)
    os.makedirs("models", exist_ok=True)
    model.save_model("models/outcome_xgb.json")
    json.dump(feat_cols, open("models/feature_cols.json", "w"))

    # Save metrics
    metrics_path = "models/outcome_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    try:
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, X, feature_names=feat_cols, show=False)
        plt.savefig("models/shap_summary.png")
        print("SHAP plot saved to models/shap_summary.png.")
    except Exception as e:
        print(f"Could not save SHAP summary plot: {e}")