import streamlit as st
from predictor import DiseasePredictor
from utils.pdf_generator import generate_pdf
from utils.history_reader import load_history

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
# User Input
# -----------------------
text = st.text_area(
    "📝 Describe your symptoms",
    height=180,
    placeholder="Example: I have fever, headache and chills."
)

# -----------------------
# Prediction
# -----------------------
if st.button("🔍 Predict Disease"):

    if text.strip() == "":
        st.warning("Please enter your symptoms.")
    else:
        result = predictor.predict(text)

        st.subheader("🦠 Predicted Disease")

        if result["confidence"] < 30:
            st.warning(
                "Prediction confidence is low. Please enter more symptoms or consult a healthcare professional."
            )
            st.info(f"Most likely prediction: {result['disease']}")
        else:
            st.success(result["disease"])

        st.subheader("📈 Confidence")
        st.write(f"{result['confidence']} %")

        st.subheader("🩺 Detected Symptoms")
        st.write(", ".join(result["symptoms"]))

        st.subheader("🏆 Top 3 Predictions")

        if result["top3"]:
            medals = ["🥇", "🥈", "🥉"]

            for i, disease in enumerate(result["top3"]):
                st.write(
                    f"{medals[i]} {disease['disease']} ({disease['confidence']}%)"
                )

        st.subheader("📖 Disease Description")
        st.write(result["description"])

        st.subheader("💊 Precautions")

        if result["precautions"]:
            for precaution in result["precautions"]:
                st.write(f"• {precaution}")
        else:
            st.write("No precautions available.")

        # -----------------------
        # PDF Download
        # -----------------------
        pdf_path = generate_pdf(result)

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_file,
                file_name="Disease_Report.pdf",
                mime="application/pdf"
            )

# -----------------------
# History
# -----------------------
st.divider()

st.header("📜 Prediction History")

try:
    history = load_history()
    st.dataframe(history, use_container_width=True)
except Exception:
    st.info("No prediction history available yet.")