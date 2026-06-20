import os

import docker


class DockerManager:
    def __init__(self):
        self.client = docker.from_env()
        self.image_name = "tensorflow-container:latest"
        self.host_dir = os.getcwd()

    def ensure_image_exists(self):
        try:
            self.client.images.get(self.image_name)
            print("Image exists")
        except docker.errors.ImageNotFound:
            print("Building image...")
            self.client.images.build(path=".", tag=self.image_name)

    def run_script(self, args_for_model, callback=None, module="defect_seg.train"):
        self.ensure_image_exists()
        # запускаем как модуль: рабочая директория /tf, пакет defect_seg лежит в /tf.
        # команда списком, чтобы пути с пробелами не разбивались
        command = ["python", "-m", module] + [str(a) for a in args_for_model]

        run_kwargs = dict(
            image=self.image_name,
            command=command,
            # device_requests работает и в Docker Desktop на Windows (WSL2 GPU);
            # runtime="nvidia" там не зарегистрирован, поэтому его не используем.
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            volumes={self.host_dir: {"bind": "/tf", "mode": "rw"}},
            working_dir="/tf",
            detach=True, stdout=True, stderr=True,
        )
        # uid/gid есть только на POSIX; на Windows os.getuid отсутствует
        if os.name == "posix":
            run_kwargs["user"] = f"{os.getuid()}:{os.getgid()}"

        container = self.client.containers.run(**run_kwargs)
        try:
            for line in container.logs(stream=True):
                text = line.decode("utf-8", errors="replace").rstrip()
                print(text)
                if callback:
                    callback(text)
        finally:
            container.remove(force=True)
