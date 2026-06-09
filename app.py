"""
app.py — Gradio deployment for BERT News Topic Classifier
Task 4: DevelopersHub Corporation AI/ML Internship

Run with:
    pip install gradio transformers torch
    python app.py
"""

import gradio as gr
from transformers import pipeline
import torch

# -------------------------------------------------------
# Load the saved fine-tuned model
# -------------------------------------------------------
MODEL_PATH = "./saved_model"   # path where trainer.save_model() saved it

print("Loading model...")
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1
)
print("Model loaded successfully.")

# -------------------------------------------------------
# Label descriptions shown in the UI
# -------------------------------------------------------
LABEL_DESCRIPTIONS = {
    "World"   : "International news, politics, global events",
    "Sports"  : "Sports events, scores, athletes, tournaments",
    "Business": "Finance, markets, economy, companies",
    "Sci/Tech": "Science, technology, innovation, research"
}

# Example headlines for the UI
EXAMPLES = [
    ["NASA launches new Mars rover to explore the red planet"],
    ["Stock markets surge as inflation fears ease"],
    ["Brazil wins the FIFA World Cup in dramatic final"],
    ["UN Security Council meets to discuss Middle East conflict"],
    ["Apple unveils new iPhone with AI-powered camera features"],
    ["Oil prices drop amid concerns over global demand"],
    ["Scientists discover new treatment for Alzheimer disease"],
    ["Olympics 2024 breaks viewership record worldwide"]
]


# -------------------------------------------------------
# Prediction function
# -------------------------------------------------------
def predict(headline):
    """
    Takes a news headline string and returns
    the predicted category with confidence scores.
    """
    if not headline.strip():
        return "Please enter a news headline.", {}

    # Get prediction
    result = classifier(headline)[0]
    label  = result["label"]
    score  = result["score"]

    # Get scores for all labels
    all_results = classifier(headline, top_k=None)
    scores_dict = {r["label"]: round(r["score"], 4) for r in all_results}

    # Format output message
    description = LABEL_DESCRIPTIONS.get(label, "")
    output = f"Category: {label}\nConfidence: {score*100:.1f}%\n\n{description}"

    return output, scores_dict


# -------------------------------------------------------
# Build Gradio Interface
# -------------------------------------------------------
with gr.Blocks(title="News Topic Classifier", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # News Topic Classifier — BERT
    **Fine-tuned on AG News Dataset | DevelopersHub Corporation AI/ML Internship**

    Enter a news headline below and the model will classify it into one of 4 categories:
    **World | Sports | Business | Sci/Tech**
    """)

    with gr.Row():
        with gr.Column(scale=2):
            headline_input = gr.Textbox(
                label="News Headline",
                placeholder="Enter a news headline here...",
                lines=3
            )
            predict_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=2):
            text_output = gr.Textbox(label="Prediction", lines=4)
            conf_output = gr.Label(label="Confidence Scores", num_top_classes=4)

    gr.Examples(
        examples=EXAMPLES,
        inputs=headline_input,
        label="Click an example to try it"
    )

    gr.Markdown("""
    ---
    ### Model Details
    | Parameter | Value |
    |---|---|
    | Base Model | bert-base-uncased |
    | Dataset | AG News (Hugging Face) |
    | Training Samples | 10,000 |
    | Epochs | 3 |
    | Learning Rate | 2e-5 |
    | Expected Accuracy | ~92-94% |
    """)

    predict_btn.click(
        fn=predict,
        inputs=headline_input,
        outputs=[text_output, conf_output]
    )

    headline_input.submit(
        fn=predict,
        inputs=headline_input,
        outputs=[text_output, conf_output]
    )


if __name__ == "__main__":
    demo.launch(share=True)   # share=True gives a public URL
