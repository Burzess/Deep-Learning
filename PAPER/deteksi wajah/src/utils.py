import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(image):
    try:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        face = mtcnn(image)
        if face is None:
            return None
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face)
        return embedding[0].cpu().numpy()
    except Exception as e:
        print(f"Error saat memproses wajah: {str(e)}")
        return None

def save_embeddings(data, path='./embeddings/embeddings.pkl'):
    torch.save(data, path)

def load_embeddings(path='./embeddings/embeddings.pkl'):
    try:
        return torch.load(path)
    except FileNotFoundError:
        return {}









# import numpy as np
# import pickle
# from mtcnn import MTCNN
# from keras_facenet import FaceNet

# # Load detector wajah MTCNN
# detector = MTCNN()

# # Load model FaceNet
# embedder = FaceNet()

# # Fungsi untuk mendeteksi wajah
# def extract_face(image):
#     faces = detector.detect_faces(image)
#     if len(faces) == 0:
#         return None
    
#     x, y, width, height = faces[0]['box']
#     face = image[y:y+height, x:x+width]
#     face = cv2.resize(face, (160, 160))
#     face = np.expand_dims(face, axis=0)
#     return face

# # Fungsi untuk mendapatkan face embedding
# def get_embedding(image):
#     face = extract_face(image)
#     if face is None:
#         return None
#     embedding = embedder.embeddings(face)
#     print(f"Panjang embedding: {len(embedding[0])}")
#     return embedding[0]  # Mengembalikan vektor embedding (512 dimensi)

# # Fungsi untuk menyimpan embeddings ke file
# def save_embeddings(data, filename="embeddings/test.pkl"):
#     with open(filename, "wb") as f:
#         pickle.dump(data, f)

# # Fungsi untuk memuat embeddings dari file
# def load_embeddings(filename="embeddings/test.pkl"):
#     with open(filename, "rb") as f:
#         return pickle.load(f)










