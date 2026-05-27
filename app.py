import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Page Config 
st.set_page_config(page_title="Simple ML Builder", layout="wide")
st.title("Simple Machine Learning App")
st.write("Upload a dataset and let the AI find patterns!")

# Data Upload 
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Data Preview")
    st.write(df.head())

    #  Choose Model 
    st.sidebar.header("2. Configure Model")
    mode = st.sidebar.selectbox("Select ML Type", ["Supervised (Prediction)", "Unsupervised (Grouping)"])

    if mode == "Supervised (Prediction)":
        st.header("Decision Tree Classifier")
        
        # Select columns
        features = st.multiselect("Select Features (Input)", df.columns, default=df.columns[:-1])
        target = st.selectbox("Select Target (What to predict)", df.columns, index=len(df.columns)-1)

        if st.button("Train & Predict"):
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

            # Accuracy
            acc = model.score(X_test, y_test)
            st.success(f"Model Accuracy: {acc:.2%}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            st.write("This shows how many times the model guessed right vs wrong.")
            
            fig, ax = plt.subplots()
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap='Blues')
            st.pyplot(fig)

    else:
        st.header("K-Means Clustering")
        clusters = st.sidebar.slider("Number of Groups (K)", 2, 10, 3)
        features = st.multiselect("Select Features for Grouping", df.columns, default=df.columns[:2])

        if st.button("Find Groups"):
            X = df[features]
            kmeans = KMeans(n_clusters=clusters, n_init=10)
            df['Group'] = kmeans.fit_predict(X)

            st.subheader("Cluster Results")
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=features[0], y=features[1], hue='Group', palette='viridis', ax=ax)
            st.pyplot(fig)
            st.write(df)

else:
    st.info("Please upload a CSV file in the sidebar to begin.")