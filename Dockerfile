FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
WORKDIR /tf
RUN python -m venv .venv1
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install keras
RUN pip install pandas
RUN pip install scikit-learn
