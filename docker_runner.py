import docker
import os


class DockerManager():
    def __init__(self):
        self.client = docker.from_env()
        self.image_name = "tensorflow-container:latest"
        self.host_dir = os.getcwd()

    def get_or_build_image(self):
        client = docker.from_env()
        image_name = "tensorflow-container:latest"
        try:
            client.images.get(image_name)
            print("Image already exists, using existing image")
        except docker.errors.ImageNotFound:
            print("Image not found, building from Dockerfile")
            image, _ = client.images.build(path=".", tag=image_name)
            print(f"Successfully built image {image.id}")
        return image_name


    def run_python_in_container(image_name, python_script_path):
        client = docker.from_env()

        user_id = os.getuid()
        group_id = os.getgid()
        command = ["python", f"/tf/{python_script_path}"]
        host_dir = os.getcwd()

        container_logs = client.containers.run(
            image=image_name,
            command=command,
            remove=True,
            user=f"{user_id}:{group_id}",
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            volumes={host_dir: {'bind': '/tf', 'mode': 'rw'}},
            working_dir="/tf",
            stdin_open=True,
            tty=True)

        print(container_logs.decode('utf-8'))


image_name = get_or_build_image()
run_python_in_container(
    image_name=image_name,
    python_script_path="helloworld.py")