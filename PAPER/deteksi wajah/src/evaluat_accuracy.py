import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from utils import get_embedding, load_embeddings
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine

class FaceEvalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Evaluator")
        self.root.geometry("950x650")
        self.db_embeddings = load_embeddings('./embeddings/embeddings.pkl')
        self.image_panel = None
        self.file_mode = False
        self.file_path = ""
        self.setup_ui()

    def setup_ui(self):
        frame_top = ttk.Frame(self.root)
        frame_top.pack(pady=5)

        ttk.Label(frame_top, text="Dataset / File").pack(side=tk.LEFT, padx=5)
        self.path_var = tk.StringVar()
        ttk.Entry(frame_top, textvariable=self.path_var, width=60).pack(side=tk.LEFT)

        ttk.Button(frame_top, text="Browse Folder", command=self.browse_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Upload File", command=self.browse_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_top, text="Evaluate", command=self.evaluate).pack(side=tk.LEFT, padx=5)

        self.result_text = tk.Text(self.root, height=20, width=100)
        self.result_text.pack(pady=10)

        self.canvas = tk.Label(self.root)
        self.canvas.pack(pady=5)

    def browse_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.file_mode = False
            self.path_var.set(path)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *webp")])
        if path:
            self.file_mode = True
            self.path_var.set(path)

    def predict(self, embedding, threshold=0.6):
        min_dist = float("inf")
        identity = "unknown"
        confidence = 0.0

        for name, db_emb in self.db_embeddings.items():
            dist = cosine(embedding, db_emb)
            if dist < threshold and dist < min_dist:
                min_dist = dist
                identity = name
                confidence = 1 - dist

        return identity, confidence

    def evaluate(self):
        path = self.path_var.get()
        if not os.path.exists(path):
            messagebox.showerror("Error", "Path tidak valid")
            return

        self.result_text.delete("1.0", tk.END)

        if self.file_mode:
            image = cv2.imread(path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            embedding = get_embedding(image)

            if embedding is not None:
                pred, conf = self.predict(embedding)
                self.result_text.insert(tk.END, f"🧠 Predict: {pred:<15} | Confidence: {conf:.2f}\n")

                pil_img = Image.fromarray(image_rgb)
                pil_img = pil_img.resize((300, 300))
                imgtk = ImageTk.PhotoImage(pil_img)
                self.canvas.configure(image=imgtk)
                self.canvas.image = imgtk

        else:
            y_true = []
            y_pred = []
            for label in os.listdir(path):
                folder_path = os.path.join(path, label)
                if os.path.isdir(folder_path):
                    for img_name in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, img_name)
                        image = cv2.imread(img_path)
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        embedding = get_embedding(image)

                        if embedding is not None:
                            pred, conf = self.predict(embedding)
                            y_true.append(label)
                            y_pred.append(pred)

                            self.result_text.insert(tk.END, f"🧠 Predict: {pred:<15} | Actual: {label:<15} | Confidence: {conf:.2f}\n")

                            pil_img = Image.fromarray(image_rgb)
                            pil_img = pil_img.resize((300, 300))
                            imgtk = ImageTk.PhotoImage(pil_img)
                            self.canvas.configure(image=imgtk)
                            self.canvas.image = imgtk

                            self.root.update()
                            self.root.after(500)

            if y_true:
                acc = accuracy_score(y_true, y_pred)
                self.result_text.insert(tk.END, f"\n✅ Akurasi: {acc * 100:.2f}% dari {len(y_true)} gambar\n")

# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEvalApp(root)
    root.mainloop()
