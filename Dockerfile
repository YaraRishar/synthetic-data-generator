FROM tensorflow/tensorflow:latest-gpu

# libgl1-mesa-glx переименован в libgl1 в свежих Ubuntu; ставим оба варианта на выбор
RUN apt-get update && \
    (apt-get install -y --no-install-recommends libgl1 || \
     apt-get install -y --no-install-recommends libgl1-mesa-glx) && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tf

# venv в контейнере не нужен (раньше .venv1 создавался, но пакеты ставились мимо
# него в системный python). Ставим напрямую.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir opencv-python-headless keras pandas scikit-learn

CMD ["python", "/tf/train.py"]
