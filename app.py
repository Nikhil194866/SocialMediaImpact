import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Social Media Impact AI", page_icon="📱", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "social_media_impact_model.pkl"
DATA_PATH = BASE_DIR / "Social_Media_Impact_OnStudents_clean.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

try:
    data = load_data()
except Exception:
    data = None

# ------------------------------------------------------------------ config
ORDER = ["Not at all", "Slightly", "Moderately", "Significantly"]
COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
EMOJI = ["🟢", "🟡", "🟠", "🔴"]
TIPS = [
    "Great balance. Keep notifications off during study blocks.",
    "Mild effect. Try phone-free study sessions of 25 minutes.",
    "Noticeable effect. Schedule fixed social media slots and use app limits.",
    "High effect. Cut study-time usage first, then reduce daily hours step by step.",
]

SCALES = {
    "AgeGroup": ["Below 18", "18–24", "25–34"],
    "Gender": ["Male", "Female", "Other"],
    "SocialMediaHours": ["Less than 1 hour", "1–3 hours", "More than 3 hours"],
    "UsageDuringStudy": ["Rarely", "Sometimes", "Often", "Always"],
    "Procrastination": ["Rarely", "Sometimes", "Often", "Always"],
    "TimeManagementDifficulty": ["Not at all", "Slightly", "Moderately", "Significantly"],
}
PLATFORMS = ["Facebook", "YouTube", "Instagram", "TikTok", "Snapchat", "LinkedIn", "Twitter", "Others"]
if data is not None and "MostUsedPlatform" in data.columns:
    PLATFORMS = data["MostUsedPlatform"].value_counts().index.tolist()

FEATURES = {
    "AgeGroup": ("Age group", SCALES["AgeGroup"]),
    "Gender": ("Gender", SCALES["Gender"]),
    "SocialMediaHours": ("Daily social media usage", SCALES["SocialMediaHours"]),
    "UsageDuringStudy": ("Social media during study", SCALES["UsageDuringStudy"]),
    "Procrastination": ("Procrastination", SCALES["Procrastination"]),
    "TimeManagementDifficulty": ("Time management difficulty", SCALES["TimeManagementDifficulty"]),
    "MostUsedPlatform": ("Most-used platform", PLATFORMS),
}
DEFAULTS = {k: v[1][0] for k, v in FEATURES.items()}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ------------------------------------------------------------------ helpers
def predict_proba(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        out = pd.DataFrame(model.predict_proba(df), columns=[str(c) for c in model.classes_])
    else:
        out = pd.get_dummies(pd.Series(model.predict(df)).astype(str)).astype(float)
    cols = [c for c in ORDER if c in out.columns] + [c for c in out.columns if c not in ORDER]
    return out[cols]


def expected_score(p: pd.Series) -> float:
    """Weighted position on the 0-3 impact scale (drives the gauge needle)."""
    n = len(p)
    if n < 2:
        return 0.0
    return float(sum(i * v for i, v in enumerate(p.values)) / (n - 1) * 3)


def pick(key: str, label: str, options: list):
    if hasattr(st, "pills"):
        val = st.pills(label, options, selection_mode="single", key=key)
    else:
        val = st.radio(label, options, horizontal=True, key=key)
    return val or DEFAULTS[key]


# ------------------------------------------------------------------ styling
st.markdown(
    """<style>
.block-container{padding-top:1.2rem;max-width:1150px}
.hero{background:linear-gradient(120deg,#6c5ce7,#00b8d9);color:#fff;border-radius:22px;padding:28px 30px;margin-bottom:16px}
.hero small{opacity:.9}
.hero h1{color:#fff;margin:6px 0 4px;font-size:2.3rem;padding:0}
.hero p{margin:0;opacity:.92}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
.stat{border:1px solid rgba(128,128,160,.3);border-radius:16px;padding:12px 16px;background:rgba(128,128,160,.08)}
.stat b{display:block;font-size:1.7rem;color:#6c5ce7}
.stat span{opacity:.7;font-size:.85rem}
.bar{display:grid;grid-template-columns:110px 1fr 44px;align-items:center;gap:10px;margin:8px 0;font-size:.92rem}
.track{height:12px;background:rgba(128,128,160,.25);border-radius:99px;overflow:hidden}
.fill{height:100%;border-radius:99px}
.verdict{text-align:center;margin:-4px 0 14px}
.verdict b{font-size:1.8rem;display:block}
.verdict span{opacity:.7;font-size:.9rem}
.tip{padding:12px 14px;border-radius:12px;background:rgba(128,128,160,.14);border-left:4px solid #6c5ce7;font-size:.92rem;margin-top:12px}
.wi{display:flex;gap:12px;align-items:flex-end;height:150px}
.wi div{flex:1;text-align:center;font-size:.8rem;opacity:.9}
.wi i{display:block;border-radius:8px 8px 0 0;margin-bottom:4px}
div[data-testid="stPills"] button{border-radius:99px}
</style>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ header
rows = f"{len(data):,}" if data is not None else "1,495"
cols_n = data.shape[1] if data is not None else 19
plat_n = len(PLATFORMS)

st.markdown(
    f"""<div class="hero"><small>📱 Statistical Inference &amp; Machine Learning</small>
<h1>Social Media Impact on Students</h1>
<p>Answer the survey and watch the estimated productivity impact update instantly.</p></div>
<div class="stats">
<div class="stat"><b>{rows}</b><span>Students surveyed</span></div>
<div class="stat"><b>{cols_n}</b><span>Survey questions</span></div>
<div class="stat"><b>{plat_n}</b><span>Platforms compared</span></div>
<div class="stat"><b>4</b><span>Impact levels</span></div>
</div>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ layout
left, right = st.columns([1.1, 1], gap="large")

with left:
    with st.container(border=True):
        st.subheader("Student Survey")
        answers = {k: pick(k, label, opts) for k, (label, opts) in FEATURES.items()}
        st.button("↺ Reset answers", on_click=reset_form)

# ------------------------------------------------------------------ prediction
try:
    proba = predict_proba(pd.DataFrame([answers])).iloc[0]
    top = proba.idxmax()
    top_i = ORDER.index(top) if top in ORDER else 0
    score = expected_score(proba)
    angle = -90 + score / 3 * 180

    bars = "".join(
        f'<div class="bar"><span>{c}</span><div class="track"><div class="fill" '
        f'style="width:{p * 100:.0f}%;background:{COLORS[ORDER.index(c)] if c in ORDER else "#6c5ce7"}"></div></div>'
        f"<b>{p:.0%}</b></div>"
        for c, p in proba.items()
    )

    gauge = (
        '<svg viewBox="0 0 220 125" style="width:100%;max-width:300px;display:block;margin:0 auto">'
        '<defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" stop-color="#2ecc71"/>'
        '<stop offset=".5" stop-color="#f1c40f"/><stop offset="1" stop-color="#e74c3c"/></linearGradient></defs>'
        '<path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="url(#gg)" stroke-width="16" stroke-linecap="round"/>'
        f'<g style="transform:rotate({angle}deg);transform-origin:110px 110px;transition:transform .8s">'
        '<line x1="110" y1="110" x2="110" y2="36" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>'
        '<circle cx="110" cy="110" r="8" fill="currentColor"/></g></svg>'
    )

    # what-if: vary daily usage with the real model
    hours_opts = FEATURES["SocialMediaHours"][1]
    wi_df = pd.DataFrame([{**answers, "SocialMediaHours": h} for h in hours_opts])
    wi_proba = predict_proba(wi_df)
    wi_html = ""
    for h, (_, row) in zip(hours_opts, wi_proba.iterrows()):
        s = expected_score(row)
        lab = row.idxmax()
        c = COLORS[ORDER.index(lab)] if lab in ORDER else "#6c5ce7"
        wi_html += (
            f'<div><i style="height:{20 + s / 3 * 100:.0f}px;background:{c}"></i>{h}<br><b>{lab}</b></div>'
        )

    with right:
        with st.container(border=True):
            st.subheader("Prediction Result")
            st.markdown(
                gauge
                + f'<div class="verdict"><b>{EMOJI[top_i]} {top}</b><span>Confidence {proba.max():.0%}</span></div>'
                + bars
                + f'<div class="tip">💡 {TIPS[top_i]}</div>',
                unsafe_allow_html=True,
            )
            st.subheader("What if usage changed?")
            st.markdown(f'<div class="wi">{wi_html}</div>', unsafe_allow_html=True)
            st.caption(
                "Machine learning estimate based on survey patterns, "
                "not a diagnosis or proof of causation."
            )
except Exception as e:
    with right:
        st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------------ data explorer
if data is not None and "ProductivityImpact" in data.columns:
    st.divider()
    with st.expander("📊 Explore the dataset"):
        x_col = st.selectbox(
            "Group by", list(data.columns), index=list(data.columns).index("MostUsedPlatform")
        )
        counts = pd.crosstab(data[x_col], data["ProductivityImpact"])
        counts = counts[[c for c in ORDER if c in counts.columns]]
        st.bar_chart(counts, color=COLORS[: counts.shape[1]])

st.caption("Student Social Media Impact Project | Powered by Scikit-learn and Streamlit")