# YOLO Microservice Project

This project provides a microservice that uses the YOLO model to detect objects in images. It offers a Flask-based REST API and processes image files, returning detected objects in JSON format.

## Requirements

To run this project, you need to install the following dependencies:

```bash
    pip install -r requirements.txt
```
# Installation and Usage
1.	Clone the project:

```bash
    git clone https://github.com/MehmetEker1/yolo-to-onnx
    cd yolo-microservice-project/convert
```
2.	Create and activate a virtual environment:
```
    python -m venv venv
    source venv/bin/activate  # macOS or Linux
    venv\Scripts\activate  # Windows
```
3.	Install the required dependencies:

```bash 
    pip install -r requirements.txt
```
4. Convert the YOLO model to ONNX format:
```bash
    python yolo_to_onnx.py
```

You have completed setting up and using the YOLO to ONNX Conversion Project. If you have any questions or encounter issues, please feel free to open an issue in the project repository.