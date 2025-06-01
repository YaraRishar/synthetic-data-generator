import threading
import tkinter as tk
from tkinter import filedialog, font

class VerifierGUI:
    def __init__(self):
        self.root = tk.Tk()

        self.real_path, self.synthetic_path = None, None
        tk.Misc.rowconfigure(self.root, [i for i in range(6)], weight=1)
        tk.Misc.columnconfigure(self.root, [i for i in range(2)], weight=1)
        label_font = font.Font(family="Verdana", size=12)
        entry_font = font.Font(family="Courier", size=12)
        button_font = font.Font(family="Verdana", size=15)

        self.root.title("Synth_Verifier")
        self.results_var = tk.StringVar()
        self.results_var.set("Результатов пока нет.")

        self.dataset_real_lbl = tk.Label(self.root, text="Датасет из реальных данных:", font=label_font)
        self.dataset_real_entry = tk.Entry(self.root, font=entry_font)
        self.dataset_real_btn = tk.Button(self.root, text=u"\U0001F4C2", font=button_font,
                                          bg="grey", command=lambda : self.ask_dir("real"))

        self.dataset_synthetic_lbl = tk.Label(self.root, text="Датасет для верификации:", font=label_font)
        self.dataset_synthetic_entry = tk.Entry(self.root, font=entry_font)
        self.dataset_synthetic_btn = tk.Button(self.root, text=u"\U0001F4C2", font=button_font,
                                               bg="grey", command=lambda : self.ask_dir("synthetic"))

        self.epoch_count_lbl = tk.Label(self.root, text="Количество эпох:", font=label_font)
        self.epoch_count_entry = tk.Entry(self.root, font=entry_font)

        self.batch_count_lbl = tk.Label(self.root, text="Размер batch:", font=label_font)
        self.batch_count_entry = tk.Entry(self.root, font=entry_font)

        self.test_size_lbl = tk.Label(self.root, text="Размер тестового датасета (%):", font=label_font)
        self.test_size_entry = tk.Entry(self.root, font=entry_font)

        self.results_lbl = tk.Label(self.root, textvariable=self.results_var, font=label_font)
        self.start_btn = tk.Button(self.root, text="Начать верификацию", font=font.Font(family="Verdana", size=12))

        self.dataset_real_lbl.grid(column=0, row=0, sticky="E", pady=5, padx=5)
        self.dataset_real_entry.grid(column=1, row=0, columnspan=3, sticky="W")
        self.dataset_real_btn.grid(column=2, row=0, sticky="W")

        self.dataset_synthetic_lbl.grid(column=0, row=1, sticky="E", pady=5, padx=5)
        self.dataset_synthetic_entry.grid(column=1, row=1, columnspan=3, sticky="W")
        self.dataset_synthetic_btn.grid(column=2, row=1, sticky="W")

        self.epoch_count_lbl.grid(column=0, row=2, sticky="E", pady=5, padx=5)
        self.epoch_count_entry.grid(column=1, row=2, sticky="W")

        self.batch_count_lbl.grid(column=0, row=3, sticky="E", pady=5, padx=5)
        self.batch_count_entry.grid(column=1, row=3, sticky="W")

        self.test_size_lbl.grid(column=0, row=4, sticky="E", pady=5, padx=5)
        self.test_size_entry.grid(column=1, row=4, sticky="W")

        self.results_lbl.grid(column=2, row=2, padx=10, pady=10)
        self.start_btn.grid(column=1, row=5, padx=5, pady=10)

        self.root.mainloop()

    def get_paths(self):
        real_dataset_path = self.dataset_real_entry.get()
        synthetic_dataset_path = self.dataset_synthetic_entry.get()
        return real_dataset_path, synthetic_dataset_path

    def ask_dir(self, dataset_type):
        file_path = filedialog.askdirectory()
        if dataset_type =="real":
            self.dataset_real_entry.delete(0, tk.END)
            self.dataset_real_entry.insert(0, file_path)
            return
        self.dataset_synthetic_entry.delete(0, tk.END)
        self.dataset_synthetic_entry.insert(0, file_path)

    def run_container(self):
        self.update_output("Контейнер запущен...")

        threading.Thread(
            target=self.container_mgr.run_script,
            args=("helloworld.py", self.update_output),
            daemon=True).start()

    def update_output(self, text):
        self.results_var.set(text)
        self.root.update_idletasks()

gui = VerifierGUI()
