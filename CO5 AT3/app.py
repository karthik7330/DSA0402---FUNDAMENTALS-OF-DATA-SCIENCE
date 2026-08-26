import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Dropout Risk System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Smart Student Dropout-Risk Prediction System")

st.markdown(
    """
    ### AI-Based Early Warning and Student Support System

    This system uses student academic, socioeconomic and
    accessibility information to identify students who may
    require additional support.

    **Important:** Predictions are intended as decision-support
    signals and must be reviewed by teachers or counsellors.
    """
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "student_dropout_dataset.csv"
    )

    data["Attendance"] = data["Attendance"].fillna(
        data["Attendance"].median()
    )

    data["Exam_Marks"] = data["Exam_Marks"].fillna(
        data["Exam_Marks"].median()
    )

    data["Digital_Access"] = data["Digital_Access"].fillna(
        data["Digital_Access"].mode()[0]
    )

    return data


data = load_data()


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    ml_data = data.copy()

    categorical_columns = [
        "Gender",
        "Region",
        "Income_Category",
        "Digital_Access",
        "Scholarship",
        "Previous_Dropout"
    ]

    encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        ml_data[column] = encoder.fit_transform(
            ml_data[column]
        )

        encoders[column] = encoder


    target_encoder = LabelEncoder()

    ml_data["Dropout_Risk"] = target_encoder.fit_transform(
        ml_data["Dropout_Risk"]
    )


    X = ml_data.drop(
        columns=[
            "Student_ID",
            "Dropout_Risk"
        ]
    )

    y = ml_data["Dropout_Risk"]


    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )

    model.fit(
        X,
        y
    )


    return (
        model,
        encoders,
        target_encoder,
        X.columns.tolist()
    )


model, encoders, target_encoder, feature_columns = train_model(
    data
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Student Prediction",
        "Risk Analysis",
        "Feature Importance"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Overall Student Risk Dashboard")


    total_students = len(data)

    risk_counts = data[
        "Dropout_Risk"
    ].value_counts()


    high_risk = risk_counts.get(
        "High",
        0
    )

    medium_risk = risk_counts.get(
        "Medium",
        0
    )

    low_risk = risk_counts.get(
        "Low",
        0
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Students",
        total_students
    )


    col2.metric(
        "High Risk",
        high_risk
    )


    col3.metric(
        "Medium Risk",
        medium_risk
    )


    col4.metric(
        "Low Risk",
        low_risk
    )


    st.divider()


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Student Dropout Risk Distribution"
    )


    risk_chart = pd.DataFrame({
        "Risk Level": [
            "Low",
            "Medium",
            "High"
        ],

        "Students": [
            low_risk,
            medium_risk,
            high_risk
        ]
    })


    st.bar_chart(
        risk_chart.set_index(
            "Risk Level"
        )
    )


    # --------------------------------------------------------
    # AVERAGE PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Academic Indicators"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "**Average Attendance**"
        )

        st.metric(
            "Attendance",
            f"{data['Attendance'].mean():.1f}%"
        )


    with col2:

        st.write(
            "**Average Exam Marks**"
        )

        st.metric(
            "Exam Marks",
            f"{data['Exam_Marks'].mean():.1f}"
        )


# ============================================================
# STUDENT PREDICTION
# ============================================================

elif page == "Student Prediction":

    st.header(
        "🔍 Individual Student Risk Prediction"
    )


    st.info(
        "Enter student information to generate an early-warning risk signal."
    )


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        age = st.number_input(
            "Age",
            min_value=12,
            max_value=18,
            value=15
        )


        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )


        region = st.selectbox(
            "Region",
            ["Rural", "Urban"]
        )


        attendance = st.slider(
            "Attendance (%)",
            0,
            100,
            75
        )


    with col2:

        exam_marks = st.slider(
            "Exam Marks",
            0,
            100,
            65
        )


        income = st.selectbox(
            "Income Category",
            [
                "Low",
                "Medium",
                "High"
            ]
        )


        distance = st.number_input(
            "Distance to School (km)",
            min_value=0.5,
            max_value=30.0,
            value=5.0
        )


    with col3:

        digital_access = st.selectbox(
            "Digital Access",
            [
                "Yes",
                "No"
            ]
        )


        scholarship = st.selectbox(
            "Scholarship Support",
            [
                "Yes",
                "No"
            ]
        )


        teacher_ratio = st.number_input(
            "Teacher-Student Ratio",
            min_value=10.0,
            max_value=60.0,
            value=30.0
        )


        previous_dropout = st.selectbox(
            "Previous Dropout",
            [
                "Yes",
                "No"
            ]
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Dropout Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "Age": [age],

            "Gender": [
                gender
            ],

            "Region": [
                region
            ],

            "Attendance": [
                attendance
            ],

            "Exam_Marks": [
                exam_marks
            ],

            "Income_Category": [
                income
            ],

            "Distance_to_School": [
                distance
            ],

            "Digital_Access": [
                digital_access
            ],

            "Scholarship": [
                scholarship
            ],

            "Teacher_Student_Ratio": [
                teacher_ratio
            ],

            "Previous_Dropout": [
                previous_dropout
            ]
        })


        # Encode input

        for column in [
            "Gender",
            "Region",
            "Income_Category",
            "Digital_Access",
            "Scholarship",
            "Previous_Dropout"
        ]:

            input_data[column] = (
                encoders[column]
                .transform(
                    input_data[column]
                )
            )


        prediction = model.predict(
            input_data[
                feature_columns
            ]
        )[0]


        probabilities = model.predict_proba(
            input_data[
                feature_columns
            ]
        )[0]


        risk = target_encoder.inverse_transform(
            [prediction]
        )[0]


        confidence = (
            probabilities[prediction]
            * 100
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Predicted Risk",
                risk
            )


        with col2:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.1f}%"
            )


        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader(
            "Recommended Support"
        )


        recommendations = []


        if risk == "High":

            recommendations.append(
                "Immediate teacher/counsellor review"
            )


            if attendance < 60:

                recommendations.append(
                    "Attendance monitoring and intervention"
                )


            if exam_marks < 40:

                recommendations.append(
                    "Additional academic support"
                )


            if income == "Low" and scholarship == "No":

                recommendations.append(
                    "Scholarship or financial-support assessment"
                )


            if digital_access == "No":

                recommendations.append(
                    "Digital learning support"
                )


            if distance > 10:

                recommendations.append(
                    "Transport/accessibility support"
                )


        elif risk == "Medium":

            recommendations.append(
                "Regular teacher monitoring"
            )


            if attendance < 75:

                recommendations.append(
                    "Attendance improvement plan"
                )


            if exam_marks < 60:

                recommendations.append(
                    "Additional academic support"
                )


            if scholarship == "No":

                recommendations.append(
                    "Check scholarship eligibility"
                )


        else:

            recommendations.append(
                "Continue regular monitoring"
            )

            recommendations.append(
                "Encourage academic participation"
            )


        for recommendation in recommendations:

            st.write(
                "• " + recommendation
            )


        st.warning(
            "This prediction should support professional "
            "judgement and should not be used to stigmatize "
            "or penalize a student."
        )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.header(
        "📈 Student Risk Analysis"
    )


    st.subheader(
        "Risk Distribution"
    )


    risk_distribution = (
        data[
            "Dropout_Risk"
        ]
        .value_counts()
        .to_frame(
            "Number of Students"
        )
    )


    st.dataframe(
        risk_distribution,
        use_container_width=True
    )


    st.subheader(
        "Average Attendance by Risk"
    )


    attendance_analysis = (
        data.groupby(
            "Dropout_Risk"
        )["Attendance"]
        .mean()
        .round(2)
    )


    st.bar_chart(
        attendance_analysis
    )


    st.subheader(
        "Average Exam Marks by Risk"
    )


    marks_analysis = (
        data.groupby(
            "Dropout_Risk"
        )["Exam_Marks"]
        .mean()
        .round(2)
    )


    st.bar_chart(
        marks_analysis
    )


    st.subheader(
        "Risk by Region"
    )


    region_analysis = pd.crosstab(
        data["Region"],
        data["Dropout_Risk"]
    )


    st.dataframe(
        region_analysis,
        use_container_width=True
    )


    st.subheader(
        "Risk by Income Category"
    )


    income_analysis = pd.crosstab(
        data["Income_Category"],
        data["Dropout_Risk"]
    )


    st.dataframe(
        income_analysis,
        use_container_width=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    st.header(
        "🎯 Factors Influencing Dropout Risk"
    )


    importance = pd.DataFrame({

        "Feature":
            feature_columns,

        "Importance":
            model.feature_importances_

    })


    importance = importance.sort_values(
        "Importance",
        ascending=False
    )


    st.dataframe(
        importance,
        use_container_width=True
    )


    st.subheader(
        "Feature Importance Chart"
    )


    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )


    st.info(
        "Feature importance indicates which variables "
        "the Random Forest model used most strongly when "
        "making predictions. It does not prove that a factor "
        "causes dropout."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Student Dropout-Risk Prediction System | "
    "CO5 Data-Driven Solution"
)