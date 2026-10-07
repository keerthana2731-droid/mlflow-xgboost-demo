import gradio as gr
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Prediction function
def predict(feature1, feature2):
    pred = model.predict([[feature1, feature2]])
    return f"Predicted Class: {pred[0]}"

# UI
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Feature 1"),
        gr.Number(label="Feature 2")
    ],
    outputs="text",
    title="XGBoost Model Prediction",
    description="Enter two numeric features to see model prediction."
)

iface.launch()