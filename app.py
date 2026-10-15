from pathlib import Path
import math
import warnings

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).with_name("Titanic-Dataset.csv")
MODEL_PATH = Path(__file__).with_name("lregression.pkl")

st.set_page_config(
    page_title="Titanic Survival Model",
    page_icon="🚢",
    layout="wide",
)


@st.cache_data
def load_passengers(path: str) -> pd.DataFrame:
    passengers = pd.read_csv(path)
    required_columns = {
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    }
    missing_columns = required_columns.difference(passengers.columns)
    if missing_columns:
        raise ValueError(
            "The Titanic dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    return passengers


@st.cache_resource
def load_model(path: str) -> tuple[object, str | None]:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = joblib.load(path)
    compatibility_warning = next(
        (
            str(warning.message)
            for warning in caught_warnings
            if issubclass(warning.category, InconsistentVersionWarning)
        ),
        None,
    )
    return model, compatibility_warning


def make_model_features(passengers: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    prepared = passengers.copy()
    prepared["Age"] = prepared["Age"].fillna(prepared["Age"].mean())
    prepared["Embarked"] = prepared["Embarked"].fillna(
        prepared["Embarked"].mode().iloc[0]
    )
    prepared["family_size"] = prepared["SibSp"] + prepared["Parch"] + 1
    prepared["Isalone"] = (prepared["family_size"] == 1).astype(int)
    prepared = pd.get_dummies(
        prepared,
        columns=["Sex", "Embarked"],
        drop_first=True,
        dtype=int,
    )
    labels = prepared["Survived"].astype(int)
    features = prepared.drop(
        columns=[
            "Survived",
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin",
        ],
        errors="ignore",
    )
    return features, labels


if not DATA_PATH.is_file():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()
if not MODEL_PATH.is_file():
    st.error(f"Saved model file not found: {MODEL_PATH}")
    st.stop()

try:
    passengers = load_passengers(str(DATA_PATH))
    model, model_warning = load_model(str(MODEL_PATH))
    features, labels = make_model_features(passengers)
except (OSError, pd.errors.ParserError, ValueError, IndexError) as error:
    st.error(f"Could not load the Titanic model or dataset: {error}")
    st.stop()

model_features = list(getattr(model, "feature_names_in_", []))
if not model_features:
    st.error(
        "The saved model does not include feature names, so safe input alignment "
        "cannot be confirmed."
    )
    st.stop()
missing_features = set(model_features).difference(features.columns)
if missing_features:
    st.error(
        "The dataset preprocessing does not produce the model's required inputs: "
        + ", ".join(sorted(missing_features))
    )
    st.stop()

features = features.loc[:, model_features]

try:
    _, X_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
    )
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)[:, 1]
except (AttributeError, TypeError, ValueError) as error:
    st.error(f"The saved model could not score the prepared Titanic data: {error}")
    st.stop()

st.title("Titanic: Data & Survival Model")
st.caption(
    "Explore the passenger dataset, estimate survival probabilities, and inspect "
    "the saved logistic-regression model."
)

if model_warning:
    st.warning(
        "The model was saved with a different scikit-learn version. "
        f"{model_warning} Check the saved model with its original version before "
        "using its predictions in production."
    )

overview_tab, passengers_tab, predict_tab, performance_tab, coefficients_tab = st.tabs(
    [
        "Project overview",
        "Explore passengers",
        "Predict survival",
        "Model performance",
        "Model coefficients",
    ]
)

with overview_tab:
    st.subheader("Dataset at a glance")
    total_passengers = len(passengers)
    survivors = int(passengers["Survived"].sum())
    survival_rate = passengers["Survived"].mean()
    average_age = passengers["Age"].mean()
    summary_columns = st.columns(4)
    summary_columns[0].metric("Passengers", f"{total_passengers:,}")
    summary_columns[1].metric("Survivors", f"{survivors:,}")
    summary_columns[2].metric("Survival rate", f"{survival_rate:.1%}")
    summary_columns[3].metric(
        "Average age", f"{average_age:.1f}" if pd.notna(average_age) else "—"
    )

    chart_left, chart_right = st.columns(2)
    outcome_counts = (
        passengers["Survived"]
        .map({0: "Did not survive", 1: "Survived"})
        .value_counts()
        .reindex(["Survived", "Did not survive"], fill_value=0)
        .rename("Passengers")
        .to_frame()
    )
    chart_left.markdown("#### Survival outcomes")
    chart_left.bar_chart(outcome_counts, color=["#168c82"])

    class_survival = (
        passengers.groupby("Pclass", observed=False)["Survived"]
        .mean()
        .mul(100)
        .rename("Survival rate (%)")
        .to_frame()
    )
    class_survival.index = [f"Class {int(value)}" for value in class_survival.index]
    chart_right.markdown("#### Survival rate by passenger class")
    chart_right.bar_chart(class_survival, color=["#e07a5f"])

    chart_left, chart_right = st.columns(2)
    sex_survival = (
        passengers.groupby("Sex", observed=False)["Survived"]
        .mean()
        .mul(100)
        .rename("Survival rate (%)")
        .to_frame()
    )
    chart_left.markdown("#### Survival rate by sex")
    chart_left.bar_chart(sex_survival, color=["#5875a4"])

    missing_values = (
        passengers.isna()
        .sum()
        .loc[lambda counts: counts.gt(0)]
        .sort_values(ascending=False)
        .rename("Missing values")
        .to_frame()
    )
    chart_right.markdown("#### Missing values by column")
    if missing_values.empty:
        chart_right.success("No missing values in the dataset.")
    else:
        chart_right.bar_chart(missing_values, color=["#d2a24c"])

with passengers_tab:
    st.subheader("Filter and inspect passenger records")
    filter_columns = st.columns(4)
    selected_classes = filter_columns[0].multiselect(
        "Passenger class",
        sorted(passengers["Pclass"].dropna().unique().tolist()),
        default=sorted(passengers["Pclass"].dropna().unique().tolist()),
        format_func=lambda value: f"Class {int(value)}",
    )
    selected_sexes = filter_columns[1].multiselect(
        "Sex",
        sorted(passengers["Sex"].dropna().astype(str).unique().tolist()),
        default=sorted(passengers["Sex"].dropna().astype(str).unique().tolist()),
    )
    selected_outcomes = filter_columns[2].multiselect(
        "Outcome",
        ["Survived", "Did not survive"],
        default=["Survived", "Did not survive"],
    )
    name_search = filter_columns[3].text_input("Passenger name")

    passenger_data = passengers.copy()
    passenger_data["Outcome"] = passenger_data["Survived"].map(
        {0: "Did not survive", 1: "Survived"}
    )
    passenger_data = passenger_data[
        passenger_data["Pclass"].isin(selected_classes)
        & passenger_data["Sex"].astype(str).isin(selected_sexes)
        & passenger_data["Outcome"].isin(selected_outcomes)
    ]
    if name_search:
        passenger_data = passenger_data[
            passenger_data["Name"].astype(str).str.contains(
                name_search, case=False, na=False
            )
        ]

    st.caption(f"Showing {len(passenger_data):,} of {total_passengers:,} passengers")
    if passenger_data.empty:
        st.info("No passengers match these filters. Try changing your selections.")
    else:
        filtered_survival = (
            passenger_data.groupby("Pclass", observed=False)["Survived"]
            .mean()
            .mul(100)
            .rename("Survival rate (%)")
            .to_frame()
        )
        filtered_survival.index = [
            f"Class {int(value)}" for value in filtered_survival.index
        ]
        st.markdown("#### Survival rate for filtered passengers")
        st.bar_chart(filtered_survival, color=["#168c82"])
        st.dataframe(
            passenger_data.drop(columns="Outcome"),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download filtered passengers",
            data=passenger_data.to_csv(index=False).encode("utf-8"),
            file_name="titanic_passengers_filtered.csv",
            mime="text/csv",
        )

with predict_tab:
    st.subheader("Try a passenger profile")
    st.write(
        "Enter passenger details to score them with the saved model. "
        "The prediction is an estimate, not a guarantee."
    )

    with st.form("passenger_prediction"):
        input_columns = st.columns(3)
        passenger_class = input_columns[0].selectbox(
            "Passenger class",
            [1, 2, 3],
            format_func=lambda value: f"Class {value}",
        )
        sex = input_columns[1].selectbox("Sex", ["female", "male"])
        age = input_columns[2].number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=29.0,
            step=1.0,
        )
        siblings_spouses = input_columns[0].number_input(
            "Siblings / spouses aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )
        parents_children = input_columns[1].number_input(
            "Parents / children aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )
        fare = input_columns[2].number_input(
            "Fare",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=1.0,
        )
        embarked = input_columns[0].selectbox(
            "Port of embarkation",
            ["S", "C", "Q"],
            format_func=lambda value: {
                "S": "Southampton",
                "C": "Cherbourg",
                "Q": "Queenstown",
            }[value],
        )
        submitted = st.form_submit_button("Predict survival")

    if submitted:
        family_size = siblings_spouses + parents_children + 1
        input_row = {
            "Pclass": passenger_class,
            "Age": age,
            "SibSp": siblings_spouses,
            "Parch": parents_children,
            "Fare": fare,
            "family_size": family_size,
            "Isalone": int(family_size == 1),
            "Sex_male": int(sex == "male"),
            "Embarked_Q": int(embarked == "Q"),
            "Embarked_S": int(embarked == "S"),
        }
        input_frame = pd.DataFrame([input_row]).reindex(columns=model_features)
        try:
            survival_probability = float(model.predict_proba(input_frame)[0, 1])
            prediction = int(model.predict(input_frame)[0])
        except (AttributeError, TypeError, ValueError) as error:
            st.error(f"The model could not score this passenger: {error}")
        else:
            result_columns = st.columns(2)
            result_columns[0].metric(
                "Estimated survival probability",
                f"{survival_probability:.1%}",
            )
            result_columns[1].metric(
                "Model prediction",
                "Survived" if prediction == 1 else "Did not survive",
            )
            st.progress(survival_probability)
            st.caption(
                "The probability is the model's estimated likelihood of survival "
                "for the profile entered above."
            )

with performance_tab:
    st.subheader("Holdout evaluation")
    st.caption(
        "Scores below use the saved model on the same 20% holdout split "
        "(random_state=42) used by the original notebook."
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Accuracy", f"{accuracy_score(y_test, test_predictions):.1%}"
    )
    metric_columns[1].metric(
        "Precision", f"{precision_score(y_test, test_predictions, zero_division=0):.1%}"
    )
    metric_columns[2].metric(
        "Recall", f"{recall_score(y_test, test_predictions, zero_division=0):.1%}"
    )
    metric_columns[3].metric(
        "F1 score", f"{f1_score(y_test, test_predictions, zero_division=0):.1%}"
    )
    metric_columns[4].metric("ROC AUC", f"{roc_auc_score(y_test, test_probabilities):.3f}")

    matrix_column, roc_column = st.columns(2)
    matrix_column.markdown("#### Confusion matrix")
    matrix_column.caption("Rows are actual outcomes; columns are model predictions.")
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    matrix_column.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual: did not survive", "Actual: survived"],
            columns=["Predicted: did not survive", "Predicted: survived"],
        ),
        width="stretch",
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test, test_probabilities
    )
    roc_data = pd.DataFrame(
        {
            "False positive rate": false_positive_rate,
            "True positive rate": true_positive_rate,
        }
    ).set_index("False positive rate")
    roc_column.markdown("#### ROC curve")
    roc_column.line_chart(roc_data, color=["#168c82"])
    roc_column.caption("A curve closer to the upper-left indicates better separation.")

    outcomes = pd.DataFrame(
        {
            "Actual outcome": y_test.map({0: "Did not survive", 1: "Survived"}),
            "Predicted outcome": pd.Series(test_predictions, index=y_test.index).map(
                {0: "Did not survive", 1: "Survived"}
            ),
        }
    )
    st.markdown("#### Actual vs. predicted outcomes")
    st.dataframe(
        pd.crosstab(outcomes["Actual outcome"], outcomes["Predicted outcome"]),
        width="stretch",
    )

with coefficients_tab:
    st.subheader("Learned logistic-regression coefficients")
    st.write(
        "Positive coefficients increase the model's survival log-odds; negative "
        "coefficients decrease them. Coefficient magnitudes are affected by each "
        "feature's measurement scale."
    )
    coefficients = pd.DataFrame(
        {
            "Feature": model_features,
            "Coefficient": model.coef_[0],
            "Odds ratio per unit": pd.Series(model.coef_[0]).map(
                lambda coefficient: math.exp(coefficient)
            ),
        }
    ).sort_values("Coefficient", key=lambda values: values.abs(), ascending=False)
    st.bar_chart(
        coefficients.set_index("Feature")[["Coefficient"]],
        color=["#5875a4"],
    )
    st.dataframe(
        coefficients.style.format(
            {"Coefficient": "{:.3f}", "Odds ratio per unit": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "An odds ratio above 1 indicates higher survival odds for a one-unit "
        "increase in that feature; below 1 indicates lower odds, holding other "
        "model inputs constant."
    )
