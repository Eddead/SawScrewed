## SawScrewed - Screw Counting and Anomaly Detection

SawScrewed is a Python-based application built with Streamlit that allows users to detect and classify screws from images, videos, or live camera feeds. The screws are classified into four categories:
1) Normal
2) Rust
3) Chipped
4) Bent

## Features
- Input options: image, video file, or live camera feed
- Screw detection and classification using custom-trained YOLO models
- Visual output of results with counts per class
- Optional custom training script to build your own model

## Project Structure
<pre> <code>SawScrewed/ 
  ├── detect/ 
  │ └── main.py # Streamlit app entry point 
  ├── model_used/ 
  │ └── trained_models/ # Pre-trained models provided 
  ├── training/ 
  │ └── train_yolo_rust.py # Script to train custom YOLO model 
  ├── requirements.txt 
  └── README.md</code> </pre>

## Getting Started
1) Clone the Repository
<pre> <code>git clone https://github.com/yourusername/SawScrewed.git
cd SawScrewed </code> </pre>

2) Create a Virtual Environment
<pre> <code>python -m venv vir_env
vir_env\Scripts\activate </code> </pre>

3) Install Requirements
Edit requirement.txt:
- Uncomment Torch CPU lines if you're using CPU
- Uncomment Torch GPU lines if you're using a compatible GPU
<pre> <code>pip install -r requirements.txt </code> </pre>

4) Run the App
<pre> <code>cd detect
streamlit run main.py </code> </pre>

## Training a Custom Model
To train your own YOLO model:
1) Navigate to the training/ folder.
2) Run:
<pre> <code>python train_yolo_rust.py </code> </pre>
3) Modify parameters inside the script as needed (e.g., batch size, image size).
4) Replace or add your own dataset in the appropriate folder structure.

## Notes
- Pre-trained models are available in model_used/trained_models/.
- Ensure your input data format matches what the model expects.
- Model training may require considerable GPU resources.
