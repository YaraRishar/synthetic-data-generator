import docker
import os


class DockerManager:
    def __init__(self, use_gpu=True):
        self.client = docker.from_env()
        self.use_gpu = use_gpu

        if use_gpu == "auto":
            self.use_gpu = self._check_gpu_available()

        self.image_name = f"tensorflow-container:latest-{'gpu' if self.use_gpu else 'cpu'}"
        self.host_dir = os.getcwd()

    @staticmethod
    def _check_gpu_available():
        import subprocess
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def ensure_image_exists(self):
        try:
            self.client.images.get(self.image_name)
            print(f"Image {self.image_name} exists")
        except docker.errors.ImageNotFound:
            print(f"Image didn\'t found. Building {self.image_name}...")
            tf_version = "latest-gpu" if self.use_gpu else "latest"
            self.client.images.build(
                path=".",
                tag=self.image_name,
                buildargs={"TF_VERSION": tf_version}
            )

    @staticmethod
    def _get_user_pair():
        try:
            return f"{os.getuid()}:{os.getgid()}"
        except AttributeError:
            return None

    def _to_container_path(self, path):
        if not isinstance(path, str):
            return str(path)
        try:
            int(path)
            return path
        except ValueError:
            pass
        abs_path = os.path.abspath(path)
        host_dir = os.path.abspath(self.host_dir)
        if os.path.commonpath([host_dir, abs_path]) == host_dir:
            rel = os.path.relpath(abs_path, host_dir)
            return os.path.join("/tf", rel).replace("\\", "/")
        return path

    def run_script(self, args_for_model, callback=None):
        self.ensure_image_exists()
        command = ["python", "/tf/model.py"] + [self._to_container_path(str(arg)) for arg in args_for_model]

        # real_path, synthetic_path, epoch_count, batch_count, test_size
        container_params = {
            'image': self.image_name,
            'command': command,
            'volumes': {self.host_dir: {'bind': '/tf', 'mode': 'rw'}},
            'working_dir': "/tf",
            'detach': True,
            'stdout': True,
            'stderr': True
        }

        user_pair = self._get_user_pair()
        if user_pair:
            container_params['user'] = user_pair

        if self.use_gpu:
            container_params['runtime'] = "nvidia"
            container_params['device_requests'] = [
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]

        try:
            container = self.client.containers.run(**container_params)
            for line in container.logs(stream=True):
                decoded_line = line.decode('utf-8').strip()
                print(decoded_line)
                if callback:
                    callback(decoded_line + "\n")

            # Получаем exit code
            result = container.wait()
            exit_code = result['StatusCode']

            if exit_code != 0 and callback:
                callback(f"The container terminated with an error (code: {exit_code})\n")
            elif callback:
                callback("Execution completed successfully\n")

        except docker.errors.ContainerError as e:
            error_msg = f"Container error: {e}"
            print(error_msg)
            if callback:
                callback(error_msg)
        except docker.errors.ImageNotFound as e:
            error_msg = f"Image didn\'t found: {e}"
            print(error_msg)
            if callback:
                callback(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(error_msg)
            if callback:
                callback(error_msg)
        finally:
            try:
                container.remove()
            except:
                pass
