import gradio as gr
from predictor import DiseasePredictor
from utils.pdf_generator import generate_pdf
from utils.history_reader import load_history

# -----------------------------
# Load predictor
# -----------------------------
predictor = DiseasePredictor()


# -----------------------------
# Prediction Function
# -----------------------------
def predict(text):

    result = predictor.predict(text)

    symptoms = ", ".join(result["symptoms"])

    precautions = ""

    if result["precautions"]:
        precautions = "\n".join(
            ["• " + p for p in result["precautions"]]
        )

    top3 = ""

    if result["top3"]:

        medals = ["🥇", "🥈", "🥉"]

        for i, disease in enumerate(result["top3"]):

            top3 += (
                f"{medals[i]} "
                f"{disease['disease']} "
                f"({disease['confidence']}%)\n"
            )

    return (
        result["disease"],
        f"{result['confidence']} %",
        symptoms,
        top3,
        result["description"],
        precautions,
    )


# -----------------------------
# PDF Generator
# -----------------------------
def generate_report(text):

    if text.strip() == "":
        return None

    result = predictor.predict(text)

    pdf_path = generate_pdf(result)

    return pdf_path


# -----------------------------
# Load CSS
# -----------------------------
with open("assets/style.css", "r", encoding="utf-8") as f:
    css = f.read()


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(
    title="AI Disease Prediction Assistant",
    css=css
) as demo:

    gr.Image(
        value="assets/logo.png",
        width=120,
        show_label=False,
        interactive=False,
        container=False,
    )

    gr.Markdown("""
# 🩺 AI Disease Prediction Assistant

### Enter your symptoms in natural language.

### Examples

- I have fever, headache and chills
- I have itching, skin rash and nodal skin eruptions
- I have vomiting and stomach pain

⚠ **This system is for educational purposes only. It is NOT a medical diagnosis.**
""")

    # ============================
    # Prediction Tab
    # ============================

    with gr.Tab("🩺 Prediction"):

        with gr.Row():

            # LEFT COLUMN
            with gr.Column():

                symptom_input = gr.Textbox(
                    label="📝 Describe Your Symptoms",
                    placeholder="Example:\nI have fever, headache and chills.",
                    lines=8,
                )

                predict_btn = gr.Button(
                    "🔍 Predict Disease",
                    variant="primary",
                )

                clear_btn = gr.Button(
                    "🗑 Clear"
                )

            # RIGHT COLUMN
            with gr.Column():

                disease = gr.Textbox(
                    label="🦠 Predicted Disease"
                )

                confidence = gr.Textbox(
                    label="📈 Confidence"
                )

                symptoms = gr.Textbox(
                    label="🩺 Detected Symptoms"
                )

                top3 = gr.Textbox(
                    label="🏆 Top 3 Predictions",
                    lines=5,
                )

                description = gr.Textbox(
                    label="📖 Disease Description",
                    lines=6,
                )

                precautions = gr.Textbox(
                    label="💊 Precautions",
                    lines=6,
                )

                download_btn = gr.Button(
                    "📄 Generate PDF Report"
                )

                pdf_file = gr.File(
                    label="📄 Download Report"
                )

    # ============================
    # History Tab
    # ============================

    with gr.Tab("📜 History"):

        history_table = gr.Dataframe(
            value=load_history(),
            interactive=False,
            wrap=True,
            label="Prediction History"
        )

        refresh_btn = gr.Button("🔄 Refresh History")

        refresh_btn.click(
            fn=load_history,
            outputs=history_table
        )

    # ============================
    # Button Events
    # ============================

    predict_btn.click(
        fn=predict,
        inputs=symptom_input,
        outputs=[
            disease,
            confidence,
            symptoms,
            top3,
            description,
            precautions,
        ],
    )

    download_btn.click(
        fn=generate_report,
        inputs=symptom_input,
        outputs=pdf_file,
    )

    clear_btn.click(
        lambda: ("", "", "", "", "", "", ""),
        outputs=[
            symptom_input,
            disease,
            confidence,
            symptoms,
            top3,
            description,
            precautions,
        ],
    )

demo.launch()