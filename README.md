# Face-Recognition-based-Attendance-and-Monitoring-System-with-Voice-Notifications

### Overview
This project is a Face Recognition-Based Attendance and Monitoring System that leverages Machine Learning (ML) and Deep Learning (DL) models for accurate face detection and recognition. It integrates Redis for efficient database management and Streamlit for an interactive web interface. The system also includes voice notifications for seamless user experience.

### →Features
•Face Detection and Recognition: Utilizes ML and DL models with InsightFace for robust and accurate face detection and recognition.

•Attendance Management: Automatically records attendance by recognizing faces, ensuring accurate and efficient tracking.

•Monitoring System: Monitors multiple faces in real-time and logs entries with timestamps.

•Voice Notifications: Provides voice-based feedback for user interactions, enhancing usability.

•Web Interface: User-friendly interface built with Streamlit for easy interaction and real-time monitoring.

•Database Management: Redis is used as the DBMS tool for fast and scalable data storage and retrieval.

### →Technologies Used
•Machine Learning (ML) & Deep Learning (DL): For developing the face detection and recognition models.

•InsightFace: A popular open-source 2D&3D deep face analysis toolbox, implemented on PyTorch and Gluon frameworks.

•Redis: A powerful in-memory data structure store used as a database for storing user data and attendance logs.

•Streamlit: A lightweight, open-source framework for building and deploying web applications in Python.

•Python: The core programming language used for developing the models and the system.


### →Installation
Prerequisites

•Python 3.8+

•Redis

•Streamlit

•Insightface

•onnxruntime

### →Usage
1.Register Faces:
Use the web interface to register new users by capturing their facial data.
Store the user data in Redis for future recognition.

2.Mark Attendance:
The system automatically recognizes registered users and marks their attendance.
Voice notifications provide feedback on successful attendance marking.
Monitor Users:

3.Real-time monitoring of faces is displayed on the web interface, allowing you to see who is present.







This Project was part of a internship project, the code here is limited to whole ML, Deep Learning and Database Management part. The Web-interface code is not attached due to some restrictions.
