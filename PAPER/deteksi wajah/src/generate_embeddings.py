import os
import cv2
import numpy as np
from utils import get_embedding, save_embeddings

DATASET_PATH = "dataset/"
embeddings = {}


for user in os.listdir(DATASET_PATH):
    user_path = os.path.join(DATASET_PATH, user)
    if os.path.isdir(user_path):  
        user_embeddings = [] 
        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            image = cv2.imread(img_path)
            embedding = get_embedding(image)
            if embedding is not None: 
                user_embeddings.append(embedding)
        
        if len(user_embeddings) > 0: 
            embeddings[user] = np.mean(user_embeddings, axis=0)
            print(embeddings[user])

# Simpan embeddings
save_embeddings(embeddings)
print(f"✅ {len(embeddings)} users' face embeddings saved.")
