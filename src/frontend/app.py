import streamlit as st, requests, json, os
import matplotlib.pyplot as plt
import numpy as np

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LexMind — Legal Research AI",
    page_icon="⚖️", layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚖️ LexMind")
    st.caption("AI-powered legal research for Indian courts")
    jurisdiction = st.selectbox(
        "Court jurisdiction",
        ["Supreme Court", "High Court", "District Court", "Tribunal"]
    )
    year_from = st.slider("Cases from year", 1990, 2024, 2000)
    st.divider()
    st.caption("Powered by LangChain · InLegalBERT · Pinecone · XGBoost · BART")

# ── Input ──────────────────────────────────────────────────────
st.title("⚖️ LexMind Legal Research")
st.markdown("Paste a case brief or legal query below.")

case_brief = st.text_area(
    "Case brief or legal question",
    height=180,
    placeholder="E.g. My client is accused under Section 420 IPC for cheating. "
               "No prior convictions. Delhi High Court. What precedents apply?"
)

analyze_btn = st.button("🔍 Analyze Case", type="primary", use_container_width=True)

# ── Analysis ───────────────────────────────────────────────────
if analyze_btn and case_brief.strip():
    with st.spinner("Analyzing — running 4 AI models..."):
        try:
            resp = requests.post(
                f"{API_URL}/analyze",
                json={"case_brief": case_brief,
                      "jurisdiction": jurisdiction,
                      "year_from": year_from},
                timeout=60
            )
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} — {resp.text}")
                st.stop()
            data = resp.json()
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            st.stop()

    st.divider()

    # ── 3-column layout ────────────────────────────────────────
    col_left, col_mid, col_right = st.columns([2, 1.2, 1])

    with col_left:
        st.subheader("📝 Case Summary")
        st.info(data.get("summary", "No summary generated."))

        st.subheader("🔍 AI Answer + Precedents")
        st.write(data.get("answer", "No answer generated."))

        st.subheader("📚 Retrieved Precedents")
        precedents = data.get("precedents", [])
        if precedents:
            for i, src in enumerate(precedents, 1):
                with st.expander(f"{i}. {src.get('citation', 'Unknown')} ({src.get('year', '')}) — {src.get('court', '')}"):
                    st.write(src.get("preview", ""))
        else:
            st.write("No precedents retrieved.")

    with col_mid:
        st.subheader("📊 Outcome Prediction")
        outcome = data.get("outcome", {})
        if outcome:
            label = outcome.get("label", "Unknown")
            confidence = outcome.get("confidence", 0.0)
            color = {"Appeal Allowed": "🟢", "Appeal Dismissed": "🔴", "Partly Allowed": "🟡"}
            st.metric(
                "Predicted verdict",
                f"{color.get(label, '⚪')} {label}",
                f"{confidence*100:.1f}% confidence"
            )
            st.progress(float(confidence))

            # Probability breakdown
            st.caption("All outcome probabilities")
            probs = outcome.get("probabilities", {})
            for lbl, prob in probs.items():
                st.write(f"**{lbl}**: {prob*100:.1f}%")
                st.progress(float(prob))

            # SHAP waterfall chart
            st.subheader("🔬 Why this prediction? (SHAP)")
            factors = outcome.get("top_factors", [])
            if factors:
                feat_names = [f.get("feature", "") for f in factors]
                shap_vals  = [f.get("shap", 0.0) for f in factors]
                colors = ["#2d7d46" if v > 0 else "#c0392b" for v in shap_vals]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(feat_names, shap_vals, color=colors)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP value (impact)", fontsize=9)
                ax.tick_params(labelsize=8)
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No SHAP factors calculated.")
        else:
            st.write("No outcome prediction available.")

    with col_right:
        st.subheader("🏷️ Extracted Entities")
        entities = data.get("entities", {})
        if entities:
            entity_colors = {
                "PETITIONER": "🔵", "RESPONDENT": "🟠",
                "JUDGE": "🟣", "STATUTE": "🟤",
                "SECTION": "⚫", "VERDICT": "🟢",
                "DATE": "📅", "COURT": "🏛️",
            }
            for label, values in entities.items():
                if values:
                    icon = entity_colors.get(label, "📌")
                    st.write(f"**{icon} {label}**")
                    for v in values[:3]:  # show top 3 per entity
                        st.caption(f"  → {v}")
        else:
            st.write("No entities extracted.")
