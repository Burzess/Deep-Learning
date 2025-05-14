import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
from utils import get_embedding, load_embeddings
from scipy.spatial.distance import cosine

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        
        self.db_embeddings = load_embeddings('./embeddings/embeddings.pkl')
        
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        self.create_widgets()
        
        self.current_image = None
        self.photo = None
        
    def create_widgets(self):
        title_label = ttk.Label(
            self.main_frame, 
            text="Face Recognition System",
            font=("Helvetica", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=10)
        
        self.image_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="solid")
        self.image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.result_label = ttk.Label(
            self.result_frame,
            text="Result: ",
            font=("Helvetica", 14, "bold")
        )
        self.result_label.grid(row=0, column=0, padx=5, pady=5)
        
        confidence_frame = ttk.Frame(self.result_frame)
        confidence_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.confidence_label = ttk.Label(
            confidence_frame,
            text="Confidence: ",
            font=("Helvetica", 12)
        )
        self.confidence_label.grid(row=0, column=0, padx=5)
        
        self.confidence_percentage = ttk.Label(
            confidence_frame,
            text="0%",
            font=("Helvetica", 12, "bold")
        )
        self.confidence_percentage.grid(row=0, column=1, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            confidence_frame,
            variable=self.progress_var,
            maximum=100,
            length=200,
            mode='determinate'
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        self.select_button = ttk.Button(
            button_frame,
            text="Select Image",
            command=self.select_image
        )
        self.select_button.grid(row=0, column=0, padx=5)
        
        self.recognize_button = ttk.Button(
            button_frame,
            text="Recognize Face",
            command=self.recognize_face
        )
        self.recognize_button.grid(row=0, column=1, padx=5)
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")
            ]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
            
    def display_image(self, cv_img):
        if cv_img is None:
            return
            
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        height, width = rgb_img.shape[:2]
        max_size = 400
        scale = min(max_size/width, max_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_img = cv2.resize(rgb_img, (new_width, new_height))
        
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
        self.image_label.configure(image=self.photo)
        
    def predict(self, embedding, threshold=0.6):
        min_dist = float("inf")
        identity = "unknown"
        for name, db_emb in self.db_embeddings.items():
            dist = cosine(embedding, db_emb)
            if dist < threshold and dist < min_dist:
                min_dist = dist
                identity = name
        return identity, min_dist
        
    def recognize_face(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        embedding = get_embedding(self.current_image)
        if embedding is None:
            messagebox.showerror("Error", "No face detected in the image!")
            return
            
        identity, confidence = self.predict(embedding)
        confidence_score = (1 - confidence) * 100  # Convert distance to confidence score
        
        self.result_label.configure(text=f"Result: {identity}")
        self.confidence_percentage.configure(text=f"{confidence_score:.2f}%")
        self.progress_var.set(confidence_score)
        
        if confidence_score >= 80:
            self.progress_bar.configure(style='green.Horizontal.TProgressbar')
        elif confidence_score >= 60:
            self.progress_bar.configure(style='yellow.Horizontal.TProgressbar')
        else:
            self.progress_bar.configure(style='red.Horizontal.TProgressbar')

def main():
    root = tk.Tk()
    
    style = ttk.Style()
    style.configure('green.Horizontal.TProgressbar', background='green')
    style.configure('yellow.Horizontal.TProgressbar', background='orange')
    style.configure('red.Horizontal.TProgressbar', background='red')
    
    app = FaceRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 