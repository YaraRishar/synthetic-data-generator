import docker
import os


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

    def run_script(self, args_for_model, callback=None):
        command = ["python", f"/tf/model.py"] + args_for_model
        command = " ".join(command)
        # real_path, synthetic_path, epoch_count, batch_count, test_size
        container = self.client.containers.run(
            image=self.image_name,
            command=command,
            user=f"{os.getuid()}:{os.getgid()}",
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            volumes={self.host_dir: {'bind': '/tf', 'mode': 'rw'}},
            working_dir="/tf",
            detach=True,
            stdout=True,
            stderr=True
        )

        for line in container.logs(stream=True):
            print(line)
            if callback:
                callback(line.decode('utf-8').strip())

        container.remove()