
import gradio as gr
import pandas as pd
import joblib
import os

# --- 1. Define Constants and Load Assets ---

# Define the paths for the model and the dataset
MODEL_PATH = 'model_pipeline.joblib'
DATASET_PATH = 'user_behavior_dataset 500 wala.csv'

# --- 2. Load the Pre-trained Model and Data ---

# Load the trained model pipeline. If it doesn't exist, the app will show an error.
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run the main training script first to create it.")
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully from model_pipeline.joblib")

    # Load the dataset just to get the list of unique device models for the dropdown
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}. It's needed for the device model list.")
    df = pd.read_csv(DATASET_PATH)
    device_models = sorted(df['Device Model'].unique())
    print("Device list loaded successfully.")

except Exception as e:
    print(f"FATAL ERROR: {e}")
    # Set placeholders if loading fails, so the app can still launch and show an error
    model_pipeline = None
    device_models = ["Error: Model or data file not found."]


# --- 3. Create the Prediction Function ---

def predict_battery_drain(device_model, screen_on_time, app_usage, num_apps, data_usage, age):
    """
    This function takes user inputs from the Gradio interface,
    processes them using the loaded model pipeline, and returns a prediction.
    """
    if model_pipeline is None:
        return "ERROR: Model is not loaded. Please check the application logs."
        
    # Create a pandas DataFrame from the user inputs.
    # The column names MUST EXACTLY MATCH the ones used for training the pipeline.
    input_data = pd.DataFrame({
        'Device Model': [device_model],
        'Screen On Time (hours/day)': [screen_on_time],
        'App Usage Time (min/day)': [app_usage],
        'Number of Apps Installed': [num_apps],
        'Data Usage (MB/day)': [data_usage],
        'Age': [age]
    })
    
    # Use the loaded pipeline to make a prediction. The pipeline handles all preprocessing.
    try:
        prediction = model_pipeline.predict(input_data)[0]
        # Format the output to be user-friendly
        return f"{prediction:.2f} mAh/day"
    except Exception as e:
        return f"Prediction Error: {e}"


# --- 4. Build the Gradio User Interface (Normal Version) ---

with gr.Blocks(title="Battery Drain Predictor") as demo:
    # Header
    gr.Markdown("# Mobile Battery Drain Predictor")
    gr.Markdown("An intelligent tool to forecast your phone's daily battery consumption based on your usage habits.")
    
    with gr.Row():
        # Input Column
        with gr.Column(scale=3):
            gr.Markdown("## Your Usage Profile")
            device_model_input = gr.Dropdown(
                choices=device_models,
                label="Device Model",
                value=device_models[0] if device_models else None,
                interactive=True
            )
            screen_time_input = gr.Slider(minimum=0, maximum=24, value=5.5, step=0.5, label="Screen On Time (hours/day)")
            app_usage_input = gr.Slider(minimum=0, maximum=1440, value=180, step=10, label="Apps Usage (min/day)")
            apps_installed_input = gr.Slider(minimum=10, maximum=250, value=70, step=1, label="Number of Apps Installed")
            data_usage_input = gr.Slider(minimum=0, maximum=5000, value=500, step=50, label="Data Usage (MB/day)")
            age_input = gr.Slider(minimum=0, maximum=72, value=12, step=1, label="Age of Phone (months)")
            
            predict_button = gr.Button("Predict Drain")

        # Output Column
        with gr.Column(scale=2):
            gr.Markdown("## Predicted Consumption")
            output_text = gr.Textbox(
                label="Predicted Battery Drain",
                interactive=False,
                placeholder="Prediction will appear here..."
            )
            gr.Markdown(
                """
                **Note:** This prediction is based on a machine learning model trained on a sample dataset.
                Actual results may vary.
                """
            )

    # Define the action for the button click
    inputs_list = [
        device_model_input,
        screen_time_input,
        app_usage_input,
        apps_installed_input,
        data_usage_input,
        age_input
    ]
    predict_button.click(fn=predict_battery_drain, inputs=inputs_list, outputs=output_text)


# --- 5. Launch the App ---
if __name__ == "__main__":
    demo.launch(debug=True)
