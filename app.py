# app.py
import re
from collections import Counter

import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from groq import Groq


# ---------------------------
# Helpers
# ---------------------------
def auto_label(texts, top_n=3):
    stop = set("""
    the a an and or but if then else when where what why how is are was were be been being
    i you we they he she it my your our their to of in on for with at from by as this that
    please add app keeps keep cant can't dont don't would amazing during using got
    """.split())

    tokens = []
    for t in texts:
        t = re.sub(r"[^a-zA-Z\s]", " ", str(t).lower())
        words = [w for w in t.split() if len(w) > 2 and w not in stop]
        tokens.extend(words)

    if not tokens:
        return "General feedback"

    common = [w for w, _ in Counter(tokens).most_common(top_n)]
    return " / ".join([w.title() for w in common])


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def embed_texts(texts):
    model = load_model()
    return model.encode(texts, show_progress_bar=False)


def generate_prd_with_groq(api_key: str, feature_name: str, reach: int, quotes: list[str]) -> str:
    client = Groq(api_key=api_key)

    prompt = f"""
You are a Product Manager. Create a concise PRD for the feature/theme below.

Feature/Theme: {feature_name}
Reach (mentions): {reach}

User quotes:
{chr(10).join(["- " + q for q in quotes])}

PRD format:
1. Problem
2. Target users
3. Goals (bullet points)
4. Non-goals
5. User stories (3-5)
6. Requirements (must-have / nice-to-have)
7. Success metrics (quantified)
8. Risks & mitigations
9. Open questions

Keep it clear, practical, and structured. Avoid fluff.
"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def safe_parse_date(series: pd.Series):
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.Series([pd.NaT] * len(series))


# ---------------------------
# UI: Page config + header
# ---------------------------
st.set_page_config(page_title="AI Feature Prioritization Copilot", page_icon="📌", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 14px; }
      .small-muted { opacity: 0.75; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📌 AI Feature Prioritization Copilot")
st.markdown('<div class="small-muted">Upload feedback → AI clusters themes → RICE prioritization → auto PRD (Groq)</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload feedback CSV", type=["csv"])
    st.caption("CSV must contain a `text` column. Optional: `date`, `source`, `id`.")
    st.divider()
    max_k_hint = st.number_input("Max themes shown (cap)", min_value=3, max_value=20, value=10)
    st.caption("Tip: Start with 4–6 themes for small datasets.")
    st.divider()
    st.subheader("Groq PRD")
    api_key = st.text_input("Groq API Key", type="password", help="Key stays in this browser session only.")

if uploaded is None:
    st.info("Upload a CSV to begin. Use your `feedback.csv` sample if needed.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = [c.lower().strip() for c in df.columns]

if "text" not in df.columns:
    st.error("Your CSV must include a column named `text`.")
    st.stop()

df["text"] = df["text"].astype(str).fillna("").str.strip()
df = df[df["text"].str.len() > 0].copy()

has_date = "date" in df.columns
has_source = "source" in df.columns

if has_date:
    df["date_parsed"] = safe_parse_date(df["date"])
else:
    df["date_parsed"] = pd.NaT

# ---------------------------
# Embeddings + Clustering
# ---------------------------
with st.spinner("Creating embeddings (first time may take a minute)..."):
    embeddings = embed_texts(df["text"].tolist())

max_k = min(int(max_k_hint), len(df))
k_default = min(5, max_k) if max_k >= 5 else max_k

k = st.slider("Number of AI themes (clusters)", min_value=2, max_value=max_k, value=max(2, k_default))

kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
labels = kmeans.fit_predict(embeddings)
df["theme_id"] = labels

# Silhouette (optional)
sil = None
try:
    sil = float(silhouette_score(embeddings, labels))
except Exception:
    sil = None

# Theme counts + names
theme_counts = (
    df.groupby("theme_id")
      .size()
      .reset_index(name="mentions")
      .sort_values("mentions", ascending=False)
)

labels_map = {}
for tid in theme_counts["theme_id"].tolist():
    texts = df[df["theme_id"] == tid]["text"].tolist()
    labels_map[tid] = auto_label(texts)

theme_counts["theme_name"] = theme_counts["theme_id"].map(labels_map)
theme_counts = theme_counts[["theme_id", "theme_name", "mentions"]]

# ---------------------------
# Tabs layout
# ---------------------------
tab_overview, tab_themes, tab_prior, tab_prd = st.tabs(["📊 Overview", "🧩 Themes", "🎯 Prioritization", "🧾 PRD"])

# ---------------------------
# OVERVIEW TAB
# ---------------------------
with tab_overview:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Snapshot")
        st.dataframe(df[["text"] + (["source"] if has_source else []) + (["date"] if "date" in df.columns else [])].head(12),
                     use_container_width=True)

    with right:
        st.subheader("KPIs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Feedback Items", len(df))
        c2.metric("Themes", k)
        c3.metric("Top Theme", theme_counts.iloc[0]["theme_name"])
        c4.metric("Top Mentions", int(theme_counts.iloc[0]["mentions"]))

        if sil is not None:
            st.caption(f"Cluster quality (silhouette): **{sil:.3f}** (higher is better)")

    st.divider()

    # Charts row
    ch1, ch2 = st.columns([1.2, 1])

    with ch1:
        st.subheader("Themes by Mentions")
        chart_df = theme_counts.copy().sort_values("mentions", ascending=True)
        st.bar_chart(chart_df.set_index("theme_name")["mentions"])

    with ch2:
        if has_source:
            st.subheader("Source Breakdown")
            src = df["source"].astype(str).fillna("unknown").value_counts().head(8)
            fig, ax = plt.subplots()
            ax.pie(src.values, labels=src.index, autopct="%1.0f%%")
            ax.set_title("Top Sources")
            st.pyplot(fig, clear_figure=True)
        else:
            st.subheader("Source Breakdown")
            st.info("No `source` column found. Add `source` to your CSV to see this chart.")

    if has_date:
        st.divider()
        st.subheader("Mentions Over Time (if dates exist)")
        temp = df.dropna(subset=["date_parsed"]).copy()
        if len(temp) == 0:
            st.info("Date column exists, but values couldn’t be parsed. Use YYYY-MM-DD format.")
        else:
            temp["day"] = temp["date_parsed"].dt.date
            daily = temp.groupby("day").size().reset_index(name="mentions")
            st.line_chart(daily.set_index("day")["mentions"])

# ---------------------------
# THEMES TAB
# ---------------------------
with tab_themes:
    st.subheader("Detected Themes")
    st.dataframe(theme_counts, use_container_width=True)

    st.caption("Click a theme below to view example quotes.")
    for _, row in theme_counts.iterrows():
        tid = int(row["theme_id"])
        name = str(row["theme_name"])
        mentions = int(row["mentions"])

        with st.expander(f"{name} (Theme {tid}) — {mentions} mentions", expanded=False):
            quotes = df[df["theme_id"] == tid]["text"].head(8).tolist()
            for q in quotes:
                st.write(f"• {q}")

# ---------------------------
# PRIORITIZATION TAB (RICE)
# ---------------------------
with tab_prior:
    st.subheader("RICE Prioritization")
    st.write("Adjust sliders to rank features like a real PM: **Reach, Impact, Confidence, Effort**.")

    rice_rows = []
    for _, row in theme_counts.iterrows():
        theme_id = int(row["theme_id"])
        theme_name = str(row["theme_name"])
        reach = int(row["mentions"])

        with st.expander(f"{theme_name} (Theme {theme_id}) — Reach: {reach}", expanded=False):
            cols = st.columns([1, 1, 1])
            with cols[0]:
                impact = st.slider("Impact", 1, 3, 2, key=f"impact_{theme_id}")
            with cols[1]:
                confidence = st.slider("Confidence", 5, 10, 8, key=f"conf_{theme_id}") / 10.0
            with cols[2]:
                effort = st.number_input("Effort (Story Points)", 1, 100, 5, key=f"effort_{theme_id}")

            rice_score = (reach * impact * confidence) / effort

            rice_rows.append({
                "theme_id": theme_id,
                "theme_name": theme_name,
                "reach": reach,
                "impact": impact,
                "confidence": round(confidence, 2),
                "effort": effort,
                "rice_score": round(rice_score, 2)
            })

    rice_df = pd.DataFrame(rice_rows).sort_values("rice_score", ascending=False)

    st.subheader("🏆 Final Ranking")
    cA, cB = st.columns([1.5, 1])
    with cA:
        st.dataframe(rice_df, use_container_width=True)

    top = rice_df.iloc[0]
    top_theme_id = int(top["theme_id"])
    top_theme_name = str(top["theme_name"])

    with cB:
        st.markdown("### Recommended Next Feature")
        st.metric("Top Feature", top_theme_name)
        st.metric("RICE Score", float(top["rice_score"]))
        st.metric("Reach (mentions)", int(top["reach"]))

    st.divider()
    st.subheader("Roadmap Export")
    st.caption("Download the ranked roadmap as CSV (shareable).")
    st.download_button(
        "⬇️ Download Ranked Roadmap (CSV)",
        data=rice_df.to_csv(index=False).encode("utf-8"),
        file_name="ranked_roadmap.csv",
        mime="text/csv",
    )

# ---------------------------
# PRD TAB
# ---------------------------
with tab_prd:
    st.subheader("PRD Generator (Groq)")
    st.write("Generate a structured PRD for the top ranked feature from the Prioritization tab.")

    st.info("Tip: Go to **🎯 Prioritization** first and adjust RICE sliders. Then come back here.")

    # Recompute top from current sliders (same as tab_prior)
    rice_rows = []
    for _, row in theme_counts.iterrows():
        theme_id = int(row["theme_id"])
        theme_name = str(row["theme_name"])
        reach = int(row["mentions"])

        impact = st.session_state.get(f"impact_{theme_id}", 2)
        conf_raw = st.session_state.get(f"conf_{theme_id}", 8)
        confidence = float(conf_raw) / 10.0
        effort = st.session_state.get(f"effort_{theme_id}", 5)

        rice_score = (reach * impact * confidence) / effort

        rice_rows.append({
            "theme_id": theme_id,
            "theme_name": theme_name,
            "reach": reach,
            "impact": impact,
            "confidence": round(confidence, 2),
            "effort": effort,
            "rice_score": round(rice_score, 2)
        })

    rice_df = pd.DataFrame(rice_rows).sort_values("rice_score", ascending=False)
    top = rice_df.iloc[0]
    top_theme_id = int(top["theme_id"])
    top_theme_name = str(top["theme_name"])
    top_reach = int(top["reach"])

    st.markdown("### Selected Feature")
    m1, m2, m3 = st.columns(3)
    m1.metric("Feature", top_theme_name)
    m2.metric("Reach", top_reach)
    m3.metric("RICE", float(top["rice_score"]))

    top_quotes = df[df["theme_id"] == top_theme_id]["text"].head(8).tolist()

    with st.expander("Quotes used for PRD", expanded=False):
        for q in top_quotes:
            st.write(f"• {q}")

    if st.button("✨ Generate PRD"):
        if not api_key:
            st.error("Paste your Groq API key in the left sidebar first.")
        else:
            try:
                with st.spinner("Generating PRD..."):
                    prd_text = generate_prd_with_groq(api_key, top_theme_name, top_reach, top_quotes)

                st.success("PRD generated!")
                st.text_area("Generated PRD", prd_text, height=520)

                st.download_button(
                    "⬇️ Download PRD (TXT)",
                    data=prd_text.encode("utf-8"),
                    file_name=f"PRD_{top_theme_name.replace(' ', '_')}.txt",
                    mime="text/plain",
                )
            except Exception as e:
                st.error(f"PRD generation failed: {e}")