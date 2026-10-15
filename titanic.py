from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "lregression.pkl"
DATA_PATH = APP_DIR / "Titanic-Dataset.csv"
FEATURE_COLUMNS = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "family_size",
    "Isalone",
    "Sex_male",
    "Embarked_Q",
    "Embarked_S",
]

st.set_page_config(
    page_title="Titanic — Survival Studio",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --ink: #172b3a;
        --muted: #71818c;
        --sea: #123c4a;
        --teal: #2f827e;
        --sand: #f4efe5;
        --line: #e7e9e5;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 86% 0%, rgba(76, 143, 135, .10), transparent 31rem),
            #fbfaf7;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #123c4a;
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #f5f2e9; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c5d4d2;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    .eyebrow {
        color: var(--teal);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .hero-title {
        color: var(--ink);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(2.5rem, 5vw, 4rem);
        line-height: 1.08;
        margin: 0;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1.04rem;
        line-height: 1.7;
        margin: .85rem 0 1.5rem;
        max-width: 690px;
    }
    .section-title {
        color: var(--ink);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.55rem;
        margin: .25rem 0 .25rem;
    }
    .section-copy { color: var(--muted); margin: 0 0 1rem; }
    .metric-card {
        background: rgba(255, 255, 255, .82);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        min-height: 102px;
        box-shadow: 0 8px 24px rgba(21, 44, 54, .035);
    }
    .metric-label {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .04em;
        text-transform: uppercase;
    }
    .metric-value {
        color: var(--ink);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: .2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #123c4a, #205e63);
        border-radius: 20px;
        color: #fff;
        margin-top: 1.35rem;
        padding: 1.5rem;
    }
    .result-card h3 { color: #fff; margin: 0 0 .4rem; }
    .result-card p { color: #d2e2df; margin: 0; }
    .stButton > button {
        background: #2f827e;
        border: 0;
        border-radius: 10px;
        color: #fff;
        font-weight: 700;
        min-height: 3rem;
        transition: background .15s ease, transform .15s ease;
    }
    .stButton > button:hover {
        background: #246b68;
        border: 0;
        color: #fff;
        transform: translateY(-1px);
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
    }
    div[data-baseweb="tab-list"] { gap: .55rem; }
    button[data-baseweb="tab"] { border-radius: 9px 9px 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_passengers():
    return pd.read_csv(DATA_PATH)


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH.name}. Run the training notebook first.")
    st.stop()
if not DATA_PATH.is_file():
    st.error(f"Passenger dataset not found: {DATA_PATH.name}.")
    st.stop()

model = load_model()
passengers = load_passengers()

with st.sidebar:
    st.markdown("## ⚓  SURVIVAL STUDIO")
    st.markdown("---")
    st.markdown("### A voyage through the data")
    st.markdown(
        "Explore the people aboard the Titanic and estimate a passenger's "
        "chance of survival with a logistic-regression model."
    )
    st.markdown("---")
    st.caption("MODEL")
    st.markdown("Logistic regression")
    st.caption("DATA")
    st.markdown(f"{len(passengers):,} passenger records")
    st.markdown("---")
    st.caption("A historical dataset, explored with care.")

st.markdown(
    f'<div class="eyebrow">A story of {len(passengers):,} lives in the manifest</div>',
    unsafe_allow_html=True,
)
st.markdown('<h1 class="hero-title">The voyage, in numbers.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-copy">Step aboard the Titanic dataset. Explore the passenger '
    'manifest, then try a survival estimate using class, family, and embarkation details.</p>',
    unsafe_allow_html=True,
)

total_passengers = len(passengers)
survival_rate = passengers["Survived"].mean() * 100
female_survival = passengers.loc[passengers["Sex"] == "female", "Survived"].mean() * 100
male_survival = passengers.loc[passengers["Sex"] == "male", "Survived"].mean() * 100

metric_columns = st.columns(4)
metrics = [
    ("Passengers", f"{total_passengers:,}"),
    ("Survived", f"{int(passengers['Survived'].sum()):,}"),
    ("Overall survival", f"{survival_rate:.1f}%"),
    ("Women survived", f"{female_survival:.1f}%"),
]
for column, (label, value) in zip(metric_columns, metrics):
    with column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

prediction_tab, insights_tab = st.tabs(["Passenger estimate", "Manifest insights"])

with prediction_tab:
    st.markdown('<div class="section-title">Build a passenger profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Adjust the details below to see what the model predicts.</p>',
        unsafe_allow_html=True,
    )

    with st.form("passenger_form"):
        left, right = st.columns(2, gap="large")
        with left:
            passenger_class = st.selectbox(
                "Passenger class",
                options=[1, 2, 3],
                index=2,
                format_func=lambda value: {
                    1: "1st class — upper deck",
                    2: "2nd class — middle deck",
                    3: "3rd class — lower deck",
                }[value],
            )
            age = st.slider("Age", min_value=0, max_value=80, value=29)
            sex = st.radio("Sex recorded in manifest", ["Female", "Male"], horizontal=True)
        with right:
            fare = st.number_input(
                "Fare paid (£)",
                min_value=0.0,
                max_value=600.0,
                value=32.0,
                step=1.0,
            )
            siblings_spouses = st.number_input(
                "Siblings / spouses aboard",
                min_value=0,
                max_value=8,
                value=0,
                step=1,
            )
            parents_children = st.number_input(
                "Parents / children aboard",
                min_value=0,
                max_value=6,
                value=0,
                step=1,
            )
            embarked = st.selectbox(
                "Port of embarkation",
                options=["Southampton", "Cherbourg", "Queenstown"],
                format_func=lambda value: {
                    "Southampton": "Southampton (S)",
                    "Cherbourg": "Cherbourg (C)",
                    "Queenstown": "Queenstown (Q)",
                }[value],
            )

        submitted = st.form_submit_button("Estimate survival", use_container_width=True)

    if submitted:
        family_size = int(siblings_spouses + parents_children + 1)
        features = pd.DataFrame(
            [[
                passenger_class,
                age,
                siblings_spouses,
                parents_children,
                fare,
                family_size,
                int(family_size == 1),
                int(sex == "Male"),
                int(embarked == "Queenstown"),
                int(embarked == "Southampton"),
            ]],
            columns=FEATURE_COLUMNS,
        )
        survived = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        survival_probability = float(probabilities[list(model.classes_).index(1)])

        if survived:
            result_title = "The model predicts survival."
            result_copy = "This profile is more likely to survive according to the fitted model."
            st.success(f"Estimated survival likelihood: {survival_probability:.1%}")
        else:
            result_title = "The model predicts non-survival."
            result_copy = "This profile is less likely to survive according to the fitted model."
            st.warning(f"Estimated survival likelihood: {survival_probability:.1%}")

        st.markdown(
            f'<div class="result-card"><h3>{result_title}</h3>'
            f'<p>{result_copy} This is a historical model estimate, not a certainty.</p></div>',
            unsafe_allow_html=True,
        )

with insights_tab:
    st.markdown('<div class="section-title">The manifest at a glance</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Survival varied across the passenger manifest. '
        'These views summarize the records included with this project.</p>',
        unsafe_allow_html=True,
    )
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.markdown("#### Survival by passenger class")
        class_rates = (
            passengers.groupby("Pclass")["Survived"]
            .mean()
            .mul(100)
            .rename("Survival rate (%)")
            .to_frame()
        )
        class_rates.index = [f"{value}{'st' if value == 1 else 'nd' if value == 2 else 'rd'} class" for value in class_rates.index]
        st.bar_chart(class_rates, color="#2f827e", y_label="Survival rate (%)")
    with chart_right:
        st.markdown("#### Survival by sex")
        sex_rates = (
            passengers.groupby("Sex")["Survived"]
            .mean()
            .mul(100)
            .rename("Survival rate (%)")
            .to_frame()
        )
        st.bar_chart(sex_rates, color="#c58b58", y_label="Survival rate (%)")

    st.markdown("#### A closer look")
    insight_columns = st.columns(3)
    insight_columns[0].metric("Women survived", f"{female_survival:.1f}%")
    insight_columns[1].metric("Men survived", f"{male_survival:.1f}%")
    insight_columns[2].metric("Average age", f"{passengers['Age'].mean():.1f} years")
    st.caption(
        "Charts describe this dataset only. The prediction model was trained on "
        "passenger class, age, fare, family size, sex, and embarkation port."
    )
