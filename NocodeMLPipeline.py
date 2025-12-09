import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------- Page config ----------
st.set_page_config(
    page_title="No-Code ML Pipeline Builder",
    layout="wide"
)

st.title("🧩 No-Code ML Pipeline Builder")
st.caption("Build a simple ML workflow with zero code: data → preprocessing → model → output")

# ---------- Helper functions for state ----------
def init_state():
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df_processed" not in st.session_state:
        st.session_state.df_processed = None
    if "target_col" not in st.session_state:
        st.session_state.target_col = None
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = None
    if "accuracy" not in st.session_state:
        st.session_state.accuracy = None

init_state()

# ---------- Pipeline flow indicator ----------
def pipeline_status():
    step1_done = st.session_state.df_raw is not None
    step2_done = st.session_state.df_processed is not None
    step3_done = st.session_state.X_train is not None
    step4_done = st.session_state.accuracy is not None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"### {'✅' if step1_done else '⬜'} 1. Upload Data")
    with col2:
        st.markdown(f"### {'✅' if step2_done else '⬜'} 2. Preprocess")
    with col3:
        st.markdown(f"### {'✅' if step3_done else '⬜'} 3. Train–Test Split")
    with col4:
        st.markdown(f"### {'✅' if step4_done else '⬜'} 4. Train Model")

pipeline_status()
st.markdown("---")

# ---------- STEP 1: Dataset Upload ----------
st.header("1️⃣ Dataset Upload")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: .csv, .xlsx, .xls"
)

if uploaded_file is not None:
    try:
        # Decide how to read file
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df_raw = df
        st.success("✅ File uploaded and read successfully!")

        # Basic dataset info
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            st.write("Column Names")
            st.write(list(df.columns))

        st.write("🔍 Preview (first 5 rows):")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Could not read the file. Please upload a valid CSV/Excel file.\n\nDetails: {e}")
else:
    st.info("Upload a CSV or Excel file to get started.")

st.markdown("---")

# ---------- Only continue if dataset is uploaded ----------
if st.session_state.df_raw is not None:
    # ---------- STEP 2: Data Preprocessing ----------
    st.header("2️⃣ Data Preprocessing")

    df = st.session_state.df_raw.copy()

    # Choose target column
    st.subheader("Target Column (Label)")
    target_col = st.selectbox(
        "Select the column you want to predict (classification target):",
        options=df.columns,
        index=None,
        placeholder="Choose target column..."
    )

    if target_col:
        st.session_state.target_col = target_col

    # Numeric columns for scaling
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    st.subheader("Feature Scaling")
    if not numeric_cols:
        st.warning("No numeric columns detected for scaling.")
        cols_to_scale = []
    else:
        cols_to_scale = st.multiselect(
            "Select numeric columns to scale:",
            options=numeric_cols,
            default=numeric_cols
        )

    scale_method = st.radio(
        "Choose a preprocessing method:",
        options=[
            "None",
            "Standardization (StandardScaler)",
            "Normalization (MinMaxScaler)"
        ],
        help="Applies only to selected numeric columns."
    )

    apply_preprocessing = st.button("Apply Preprocessing")

    if apply_preprocessing:
        df_processed = df.copy()

        if scale_method != "None" and cols_to_scale:
            if scale_method.startswith("Standardization"):
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
            st.success(f"✅ {scale_method} applied to: {cols_to_scale}")
        else:
            st.info("No scaling applied (either 'None' selected or no columns chosen).")

        st.session_state.df_processed = df_processed

        st.subheader("Processed Data Preview")
        st.dataframe(df_processed.head())

    elif st.session_state.df_processed is not None:
        # Show previously processed data as info
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state.df_processed.head())

    st.markdown("---")

    # ---------- STEP 3: Train–Test Split ----------
    st.header("3️⃣ Train–Test Split")

    if st.session_state.df_processed is None:
        st.info("Apply preprocessing first (or choose 'None') to continue.")
    elif st.session_state.target_col is None:
        st.warning("Please select a target column in Step 2 to continue.")
    else:
        df_processed = st.session_state.df_processed
        target_col = st.session_state.target_col

        if target_col not in df_processed.columns:
            st.error("The selected target column is not present in the processed data.")
        else:
            # Configure split ratio
            train_size_pct = st.slider(
                "Select train set size (%)",
                min_value=50,
                max_value=90,
                value=80,
                step=5
            )
            train_size = train_size_pct / 100.0
            test_size = 1 - train_size

            # Perform split
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]

            if y.nunique() < 2:
                st.error("The target column must have at least 2 different classes.")
            else:
                split_button = st.button("Run Train–Test Split")

                if split_button:
                    try:
                        # Try stratified split (good for classification), fall back if it fails
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y,
                                train_size=train_size,
                                random_state=42,
                                stratify=y
                            )
                        except ValueError:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y,
                                train_size=train_size,
                                random_state=42
                            )

                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test

                        st.success("✅ Train–test split completed!")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Train Set Size", f"{X_train.shape[0]} rows")
                        with c2:
                            st.metric("Test Set Size", f"{X_test.shape[0]} rows")

                    except Exception as e:
                        st.error(f"❌ Error during train–test split: {e}")

                # If already split before, show info
                if st.session_state.X_train is not None:
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Train Set Size", f"{X_train.shape[0]} rows")
                    with c2:
                        st.metric("Test Set Size", f"{X_test.shape[0]} rows")

    st.markdown("---")

    # ---------- STEP 4: Model Selection & Training ----------
    st.header("4️⃣ Model Selection & Results")

    if st.session_state.X_train is None:
        st.info("Complete the train–test split to enable model training.")
    else:
        model_choice = st.radio(
            "Choose a model to train:",
            options=[
                "Logistic Regression",
                "Decision Tree Classifier"
            ]
        )

        train_button = st.button("Train Model")

        if train_button:
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test

            try:
                if model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                else:
                    model = DecisionTreeClassifier(random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.session_state.model_name = model_choice
                st.session_state.accuracy = acc

                st.success("✅ Model trained successfully!")

                # Show results
                st.subheader("Model Performance")
                st.write(f"**Model:** {model_choice}")
                st.metric("Accuracy on Test Set", f"{acc:.3f}")

                # Confusion matrix visualization
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error while training the model: {e}")

        # If there are previous results, show them
        if st.session_state.accuracy is not None and not train_button:
            st.subheader("Last Trained Model")
            st.write(f"**Model:** {st.session_state.model_name}")
            st.metric("Accuracy on Test Set", f"{st.session_state.accuracy:.3f}")
else:
    # If no data uploaded yet, we already showed info above
    pass
