# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import altair as alt
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from Bio import SeqIO, pairwise2
# import re
# import io
# import random

# # Set up the Streamlit app
# st.set_page_config(page_title='AI Precision Medicine for NF', layout='wide')
# st.title("🔬 Shivi Kumar's AI-Powered Precision Medicine for Neurofibromatosis (NF)")
# st.markdown("AI-driven predictions for drug response, surgical outcomes, and CRISPR-based gene therapy recommendations.")

# # Sidebar for data upload
# st.sidebar.header("Upload Genomic Dataset")
# uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# def load_data(file):
#     df = pd.read_csv(file)
#     return df

# if uploaded_file:
#     df = load_data(uploaded_file)
#     st.subheader("📊 Uploaded Dataset Preview")
#     st.dataframe(df.head())

#     # Select target outcome
#     target = st.sidebar.selectbox("Select Target Variable", df.columns)

#     # Feature selection
#     features = st.sidebar.multiselect("Select Features", df.columns, default=df.columns[:-1])
    
#     if st.sidebar.button("Train AI Model"):
#         X = df[features]
#         y = df[target]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)
        
#         joblib.dump(model, 'nf_model.pkl')
#         st.success("AI Model Trained Successfully! 🎉")
        
#         # Model Evaluation
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         st.subheader("Model Performance")
#         st.write(f"**Accuracy:** {accuracy:.2f}")
#         st.text("Classification Report:")
#         st.text(classification_report(y_test, y_pred))
        
#         # Confusion Matrix Visualization
#         st.subheader("Confusion Matrix")
#         cm = confusion_matrix(y_test, y_pred)
#         fig, ax = plt.subplots()
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#         st.pyplot(fig)
        
#         st.subheader("Feature Importance")
#         feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
#         chart = alt.Chart(feature_importance).mark_bar().encode(
#             x=alt.X('Importance', title='Feature Importance'),
#             y=alt.Y('Feature', sort='-x'),
#             tooltip=['Feature', 'Importance']
#         )
#         st.altair_chart(chart, use_container_width=True)

# # Mocked CRISPR guide RNA prediction
# st.subheader("🧬 AI-Optimized CRISPR Guide RNA for NF1/NF2 Mutations")
# st.markdown("Upload patient genomic sequences for AI-powered CRISPR guide RNA recommendations.")

# def clean_sequence(sequence):
#     """Cleans and extracts valid DNA sequences from raw input or FASTA format."""
#     sequence = sequence.strip().upper()
#     if sequence.startswith(">"):
#         fasta_io = io.StringIO(sequence)
#         sequences = [str(record.seq) for record in SeqIO.parse(fasta_io, "fasta")]
#         sequence = "".join(sequences)
#     sequence = re.sub(r"[^ACGT]", "", sequence)
#     return sequence

# def mock_crispr_guides(sequence):
#     """Mocks CRISPR guide RNA generation."""
#     guide_rnas = [
#         f"Guide_{i}: {sequence[random.randint(0, len(sequence)-20):random.randint(0, len(sequence)-1)]}NGG"
#         for i in range(1, 6)
#     ]
#     return guide_rnas

# guide_rna_placeholder = st.text_area("Enter Genomic Sequence (FASTA or Raw DNA Sequence)")
# if st.button("Generate CRISPR Guide RNA"):
#     cleaned_sequence = clean_sequence(guide_rna_placeholder)
#     if cleaned_sequence:
#         guide_rnas = mock_crispr_guides(cleaned_sequence)
#         if guide_rnas:
#             st.success("⚡ AI-generated guide RNA sequences:")
#             for guide in guide_rnas:
#                 st.write(guide)
#         else:
#             st.warning("No suitable CRISPR guides found. Try a different sequence.")
#     else:
#         st.error("❌ Invalid sequence. Please enter a valid DNA sequence containing only A, C, G, T.")

# # Data Visualization Section
# st.subheader("📈 Exploratory Data Analysis")
# if uploaded_file:
#     st.subheader("Feature Distribution")
#     selected_feature = st.selectbox("Select Feature to Visualize", df.columns)
#     fig = px.histogram(df, x=selected_feature, nbins=30, title=f"Distribution of {selected_feature}")
#     st.plotly_chart(fig)
    
#     st.subheader("Correlation Matrix")
#     corr_matrix = df.corr()
#     fig, ax = plt.subplots()
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
#     st.pyplot(fig)

# st.sidebar.markdown("---")
# st.sidebar.markdown("👩‍⚕️ **Built for cutting-edge NF research and precision medicine.**")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Bio import SeqIO, pairwise2
import re
import io
import random

# Set up the Streamlit app
st.set_page_config(page_title='AI Precision Medicine for NF', layout='wide')
st.title("🔬 AI-Powered Precision Medicine for Neurofibromatosis (NF)")
st.markdown("AI-driven predictions for drug response, surgical outcomes, and CRISPR-based gene therapy recommendations.")

# Sidebar for data upload
st.sidebar.header("Upload Genomic Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("📊 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Select target outcome
    target = st.sidebar.selectbox("Select Target Variable", df.columns)

    # Feature selection
    features = st.sidebar.multiselect("Select Features", df.columns, default=df.columns[:-1])
    
    if st.sidebar.button("Train AI Model"):
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        joblib.dump(model, 'nf_model.pkl')
        st.success("AI Model Trained Successfully! 🎉")
        
        # Model Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Performance")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Confusion Matrix Visualization
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        chart = alt.Chart(feature_importance).mark_bar().encode(
            x=alt.X('Importance', title='Feature Importance'),
            y=alt.Y('Feature', sort='-x'),
            tooltip=['Feature', 'Importance']
        )
        st.altair_chart(chart, use_container_width=True)

# Mocked CRISPR guide RNA prediction with condition diagnosis
st.subheader("🧬 AI-Optimized CRISPR Guide RNA for NF1/NF2 Mutations")
st.markdown("Upload patient genomic sequences for AI-powered CRISPR guide RNA recommendations and diagnosis.")

mutation_database = {
    "NF1_Exon11": "Neurofibromatosis Type 1",
    "NF2_Exon2": "Neurofibromatosis Type 2",
    "NF1_Exon17": "Severe NF1 Phenotype",
    "NF2_Frameshift": "Schwannomatosis"
}

def clean_sequence(sequence):
  
    sequence = sequence.strip().upper()
    if sequence.startswith(">"):
        fasta_io = io.StringIO(sequence)
        sequences = [str(record.seq) for record in SeqIO.parse(fasta_io, "fasta")]
        sequence = "".join(sequences)
    sequence = re.sub(r"[^ACGT]", "", sequence)
    return sequence

def mock_crispr_guides(sequence):
    """Mocks CRISPR guide RNA generation and diagnoses patient condition."""
    guide_rnas = [
        f"Guide_{i}: {sequence[random.randint(0, len(sequence)-20):random.randint(0, len(sequence)-1)]}NGG"
        for i in range(1, 6)
    ]
    detected_condition = random.choice(list(mutation_database.values()))
    return guide_rnas, detected_condition

guide_rna_placeholder = st.text_area("Enter Genomic Sequence (FASTA or Raw DNA Sequence)")
if st.button("Generate CRISPR Guide RNA & Diagnose Condition"):
    cleaned_sequence = clean_sequence(guide_rna_placeholder)
    if cleaned_sequence:
        guide_rnas, condition = mock_crispr_guides(cleaned_sequence)
        if guide_rnas:
            st.success("⚡ AI-generated guide RNA sequences:")
            for guide in guide_rnas:
                st.write(guide)
            st.subheader("🩺 Predicted Patient Condition")
            st.write(f"The patient is likely diagnosed with: **{condition}**")
        else:
            st.warning("No suitable CRISPR guides found. Try a different sequence.")
    else:
        st.error("❌ Invalid sequence. Please enter a valid DNA sequence containing only A, C, G, T.")

# Data Visualization Section
st.subheader("📈 Exploratory Data Analysis")
if uploaded_file:
    st.subheader("Feature Distribution")
    selected_feature = st.selectbox("Select Feature to Visualize", df.columns)
    fig = px.histogram(df, x=selected_feature, nbins=30, title=f"Distribution of {selected_feature}")
    st.plotly_chart(fig)
    
    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("👩‍⚕️ **Built for cutting-edge NF research and precision medicine.**")
