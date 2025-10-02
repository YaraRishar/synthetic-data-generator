ARG TF_VERSION=latest-gpu
FROM tensorflow/tensorflow:${TF_VERSION}

# Системные зависимости лучше ставить одним слоем
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tf

# витуальная среда в docker-контейнре не нужна - он сам по себе хорошо изолирован
# RUN python -m venv .venv1

# python-пакеты тоже ставим одним слоем (можно было отправить все в requirements.txt и устанавливать уже оттуда)...
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    opencv-python \
    keras \
    pandas \
    scikit-learn

CMD ["/bin/bash"]