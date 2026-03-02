# app.py
import re
from collections import Counter

import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# NEW: Groq SDK for PRD generation
from groq import Groq


# ---------------------------
# Helpers
# ---------------------------
def auto_label(texts, top_n=3):
    """
    Quick, no-API theme label using top keywords.
    """
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
    """
    Calls Groq LLM to generate a concise PRD for the selected feature/theme.
    """
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


# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="AI Feature Prioritization Copilot", layout="wide")
st.title("AI Feature Prioritization Copilot 🚀")
st.caption("Upload user feedback → AI groups it into themes → you get prioritized features + PRD (PM-style).")

uploaded = st.file_uploader("Upload feedback CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV with at least a `text` column. (Optional: `date`, `source`, `id`.)")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = [c.lower().strip() for c in df.columns]

if "text" not in df.columns:
    st.error("Your CSV must include a column named `text`.")
    st.stop()

df["text"] = df["text"].astype(str).fillna("").str.strip()
df = df[df["text"].str.len() > 0].copy()

st.subheader("Preview")
st.dataframe(df.head(30), use_container_width=True)

st.divider()
st.subheader("1) AI Theme Clustering")
st.write("We embed each feedback item, then cluster similar ones into themes.")

# Embeddings
with st.spinner("Creating embeddings... (first time can take a minute)"):
    embeddings = embed_texts(df["text"].tolist())

# Choose number of clusters
max_k = min(10, len(df))
k = st.slider(
    "How many themes (clusters) do you want?",
    min_value=2,
    max_value=max_k,
    value=min(4, max_k),
)

# Clustering
kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
labels = kmeans.fit_predict(embeddings)
df["theme_id"] = labels

# Cluster quality metric
try:
    sil = silhouette_score(embeddings, labels)
    st.caption(f"Cluster quality (silhouette score): {sil:.3f} (higher is better)")
except Exception:
    pass

# Theme counts
theme_counts = (
    df.groupby("theme_id")
      .size()
      .reset_index(name="mentions")
      .sort_values("mentions", ascending=False)
)

# Auto names for themes
labels_map = {}
for tid in theme_counts["theme_id"].tolist():
    texts = df[df["theme_id"] == tid]["text"].tolist()
    labels_map[tid] = auto_label(texts)

theme_counts["theme_name"] = theme_counts["theme_id"].map(labels_map)

st.subheader("2) Prioritized Themes (MVP = Reach by mentions)")
st.dataframe(theme_counts[["theme_id", "theme_name", "mentions"]], use_container_width=True)

st.subheader("3) Theme Examples (quotes)")
for theme_id in theme_counts["theme_id"].tolist():
    name = labels_map.get(theme_id, f"Theme {theme_id}")
    count = int(theme_counts[theme_counts["theme_id"] == theme_id]["mentions"].iloc[0])

    st.markdown(f"### {name} (Theme {theme_id}) — {count} mentions")
    theme_df = df[df["theme_id"] == theme_id].head(5)
    for t in theme_df["text"].tolist():
        st.write(f"• {t}")

st.divider()
st.subheader("4) PM Prioritization (RICE Scoring)")
st.write("Adjust Impact, Confidence, and Effort to compute real product priorities.")

rice_rows = []

for _, row in theme_counts.iterrows():
    theme_id = int(row["theme_id"])
    theme_name = str(row["theme_name"])
    reach = int(row["mentions"])

    with st.expander(f"{theme_name} (Theme {theme_id}) — {reach} mentions", expanded=False):
        impact = st.slider(
            f"Impact (Theme {theme_id})",
            min_value=1,
            max_value=3,
            value=2,
            key=f"impact_{theme_id}"
        )

        confidence = st.slider(
            f"Confidence (Theme {theme_id})",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.1,
            key=f"conf_{theme_id}"
        )

        effort = st.number_input(
            f"Effort (Story Points) (Theme {theme_id})",
            min_value=1,
            max_value=100,
            value=5,
            key=f"effort_{theme_id}"
        )

        rice_score = (reach * impact * confidence) / effort

        rice_rows.append({
            "theme_id": theme_id,
            "theme_name": theme_name,
            "reach": reach,
            "impact": impact,
            "confidence": confidence,
            "effort": effort,
            "rice_score": round(rice_score, 2)
        })

rice_df = pd.DataFrame(rice_rows).sort_values("rice_score", ascending=False)

st.subheader("🏆 Final Priority Ranking")
st.dataframe(rice_df, use_container_width=True)

top = rice_df.iloc[0]
top_theme_id = int(top["theme_id"])
top_theme_name = str(top["theme_name"])
top_reach = int(top["reach"])

st.success(
    f"🚀 Recommended Next Feature: **{top_theme_name}** (Theme {top_theme_id}) "
    f"— RICE: {top['rice_score']}"
)

# ---------------------------
# PRD Generator (Groq)
# ---------------------------
st.divider()
st.subheader("5) PRD Generator (Groq)")
st.write("Paste your Groq API key below (kept only in this browser session).")

api_key = st.text_input("Groq API Key", type="password")

top_quotes = df[df["theme_id"] == top_theme_id]["text"].head(8).tolist()

col1, col2 = st.columns([1, 2])

with col1:
    st.write("**Top feature inputs:**")
    st.write(f"- Feature: {top_theme_name}")
    st.write(f"- Reach: {top_reach}")
    st.write("- Quotes used: 8")

with col2:
    if st.button("Generate PRD for Top Priority Feature"):
        if not api_key:
            st.error("Please paste your Groq API key first.")
        else:
            try:
                with st.spinner("Generating PRD..."):
                    prd = generate_prd_with_groq(api_key, top_theme_name, top_reach, top_quotes)
                st.text_area("Generated PRD", prd, height=520)
            except Exception as e:
                st.error(f"PRD generation failed: {e}")