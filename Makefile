.PHONY: build-gpu build-cpu build-all run-gpu run-cpu

build-gpu:
	docker build --build-arg TF_VERSION=latest-gpu -t tensorflow-container:latest-gpu .

build-cpu:
	docker build --build-arg TF_VERSION=latest -t tensorflow-container:latest-cpu .

build-all: build-gpu build-cpu

run-gui:
	python main.py