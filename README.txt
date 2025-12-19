CANCER DETECTION USING MACHINE LEARNING

Overview
--------
This project presents a hybrid cancer detection system designed to assist in the early diagnosis of
Breast Cancer and Lung Cancer using machine learning and deep learning techniques. The system supports
dual-modality input, including medical images and PDF-based clinical reports, and provides real-time
predictions through a web-based interface built using Django.

The objective of the project is to improve the accessibility, accuracy, and speed of cancer diagnosis
by combining classical machine learning and deep learning models in a single scalable platform.

Key Features
------------
- Dual input support:
  • Image-based detection using Convolutional Neural Networks (CNN)
  • Report-based detection using Logistic Regression (PDF medical reports)
- Cancer types covered:
  • Breast Cancer (Benign / Malignant)
  • Lung Cancer (Benign / Malignant)
- Web application built with Django
- Fast inference with real-time predictions
- Privacy-aware design with no permanent storage of uploaded files

System Architecture
-------------------
The system follows a modular layered architecture:
- Frontend for file upload and result visualization
- Backend for request handling, preprocessing, and inference
- Machine Learning layer with independent models selected dynamically based on input type

Machine Learning Models
-----------------------
1. Image-Based Detection (CNN)
- Framework: TensorFlow / Keras
- Input size: 224 x 224 x 3
- Architecture:
  • Convolutional layers with ReLU activation
  • MaxPooling and Dropout layers
  • Fully connected dense layers
- Output: Binary classification (Benign / Malignant)

Performance:
- Breast Cancer (Images): ~93% accuracy
- Lung Cancer (Images): ~91% accuracy

2. Report-Based Detection (Logistic Regression)
- Framework: Scikit-learn
- Input: Clinical features extracted from PDF medical reports
- Feature extraction using pdfplumber and regex
- Output: Binary classification (Benign / Malignant)

Performance:
- Report-based predictions achieved accuracy between 85% and 95% depending on data quality

Technologies Used
-----------------
- Programming Language: Python
- Machine Learning: TensorFlow, Keras, Scikit-learn
- Computer Vision: OpenCV
- PDF Processing: pdfplumber
- Web Framework: Django
- Frontend: HTML, CSS, Tailwind CSS
- Visualization: Matplotlib, Seaborn
- Development Tools: Jupyter Notebook, VS Code

Results Summary
---------------
- CNN models consistently achieved over 90% accuracy
- Logistic Regression handled incomplete report features effectively
- Image-based models outperformed report-based models in recall
- System tested with real-world samples and concurrent requests

Limitations
-----------
- Performance depends on dataset quality
- PDF parsing may fail for unstructured reports
- CNN models lack explainability
- Not integrated with hospital EHR systems

Future Enhancements
-------------------
- Explainable AI techniques (Grad-CAM, SHAP)
- Improved NLP-based PDF parsing
- Integration with hospital databases
- Mobile application support

Academic Context
----------------
This project was developed as a Final Year Engineering Project under Visvesvaraya Technological
University (VTU) at Nitte Meenakshi Institute of Technology, Bengaluru.

Live Demonstration
------------------
The system can be demonstrated live, including:
- Image upload and CNN inference
- PDF upload and feature extraction
- Real-time prediction display through the web interface

Author
------
Aman Malik
Computer Science & Engineering
Nitte Meenakshi Institute of Technology
