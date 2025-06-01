import tkinter as tk
from tkinter import scrolledtext
from test_manager import ContainerManager
import threading


class DockerGUI:
    def __init__(self, root):
        self.root = root
        self.container_mgr = ContainerManager()
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Docker Container GUI")

        # Output display
        self.output_area = scrolledtext.ScrolledText(self.root, height=20)
        self.output_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Run button
        run_btn = tk.Button(
            self.root,
            text="Run Container",
            command=self.run_container
        )
        run_btn.pack(pady=10)

    def update_output(self, text):
        """Callback to update GUI with container output"""
        self.output_area.insert(tk.END, text + "\n")
        self.output_area.see(tk.END)
        self.root.update_idletasks()

    def run_container(self):
        """Run container and display output"""
        self.output_area.delete(1.0, tk.END)
        self.output_area.insert(tk.END, "Starting container...\n")

        threading.Thread(
            target=self.container_mgr.run_script,
            args=("helloworld.py", self.update_output),
            daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = DockerGUI(root)
    root.mainloop()