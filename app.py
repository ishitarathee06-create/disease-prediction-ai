import streamlit as st

st.write("App started")

from predictor import DiseasePredictor

st.write("Predictor imported")

from utils.pdf_generator import generate_pdf

st.write("PDF imported")

from utils.history_reader import load_history

st.write("History imported")

# -----------------------
# Page Settings
# -----------------------
st.set_page_config(
    page_title="AI Disease Prediction Assistant",
    page_icon="🩺",
    layout="wide"
)

# -----------------------
# Load Predictor
# -----------------------
@st.cache_resource
def load_predictor():
    return DiseasePredictor()

predictor = load_predictor()

# -----------------------
# Logo
# -----------------------
st.image("assets/logo.png", width=120)

st.title("🩺 AI Disease Prediction Assistant")

st.markdown("""
Enter your symptoms in natural language.

### Examples

- I have fever, headache and chills
- I have itching, skin rash and nodal skin eruptions
- I have vomiting and stomach pain

⚠️ **This tool is for educational purposes only. It is NOT a medical diagnosis.**
""")

# -----------------------
# Input
# -----------------------
text = st.text_area(
    "📝 Describe your symptoms",
    height=180,
    placeholder="Example: I have fever, headache and chills."
)

if st.button("🔍 Predict Disease"):

    result = predictor.predict(text)

    st.subheader("🦠 Predicted Disease")

    if result["confidence"] < 30:
        st.warning(
            "⚠️ The prediction confidence is low. Please enter more symptoms or consult a healthcare professional."
        )
        st.info(f"Most likely prediction: {result['disease']}")
    else:
        st.success(result["disease"])

    st.subheader("📈 Confidence")
    st.write(f"{result['confidence']} %")

    st.subheader("🩺 Detected Symptoms")
    st.write(", ".join(result["symptoms"]))

    st.subheader("🏆 Top 3 Predictions")

    for disease in result["top3"]:
        st.write(
            f"• {disease['disease']} ({disease['confidence']}%)"
        )

    st.subheader("📖 Description")
    st.write(result["description"])

    st.subheader("💊 Precautions")

    for p in result["precautions"]:
        st.write("•", p)

    pdf_path = generate_pdf(result)

    with open(pdf_path, "rb") as f:
        st.download_button(
            "📄 Download PDF Report",
            data=f,
            file_name="Disease_Report.pdf",
            mime="application/pdf"
        )

st.divider()

st.header("📜 Prediction History")

history = load_history()

st.dataframe(history, use_container_width=True)
