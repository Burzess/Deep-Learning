import cv2
import numpy as np
from scipy.spatial.distance import cosine
from utils import get_embedding, load_embeddings 
from mtcnn import MTCNN

embeddings = load_embeddings()

detector = MTCNN()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height

        face_crop = frame[y:y2, x:x2]

        embedding = get_embedding(face_crop)

        if embedding is not None:
            best_match = None
            min_distance = float("inf")

            for user, stored_embedding in embeddings.items():
                distance = cosine(embedding, stored_embedding)
                if distance < 0.5 and distance < min_distance:
                    min_distance = distance
                    best_match = user

            # Jika wajah tidak dikenali, tandai sebagai "Unknown"
            if best_match is None or min_distance > 0.5:
                best_match = "Unknown"

            # Gambar bounding box dan label nama
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, best_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame dengan deteksi wajah
    cv2.imshow("Face Recognition", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







# import cv2
# import numpy as np
# from scipy.spatial.distance import cosine
# from utils import get_embedding, load_embeddings, save_embeddings
# from mtcnn import MTCNN
# import tkinter as tk
# from tkinter import simpledialog

# # Fungsi untuk input nama user lewat GUI
# def get_username():
#     root = tk.Tk()
#     root.withdraw()  # Sembunyikan main window
#     name = simpledialog.askstring("Input User Baru", "Masukkan nama user baru:")
#     root.destroy()
#     return name

# # Inisialisasi detektor wajah
# detector = MTCNN()

# # Load database embeddings
# embeddings = load_embeddings()

# # Buka kamera
# cap = cv2.VideoCapture(0)

# # Counter nama default user (jika perlu)
# user_counter = len(embeddings) + 1

# print("🚀 Sistem siap. Tekan 's' untuk simpan wajah baru, 'q' untuk keluar.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Deteksi wajah
#     faces = detector.detect_faces(frame)
#     if len(faces) == 0:
#         cv2.putText(frame, "Tidak ada wajah terdeteksi", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Face Recognition", frame)
#         key = cv2.waitKey(30) & 0xFF
#         if key == ord('q'):
#             break
#         continue

#     # Ambil wajah pertama saja
#     x, y, width, height = faces[0]['box']
#     x, y = max(0, x), max(0, y)
#     face = frame[y:y + height, x:x + width]

#     embedding = get_embedding(face)
#     if embedding is None:
#         cv2.putText(frame, "Wajah gagal diproses", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Face Recognition", frame)
#         key = cv2.waitKey(30) & 0xFF
#         if key == ord('q'):
#             break
#         continue

#     # Pencocokan dengan database
#     best_match = None
#     min_distance = float("inf")

#     for user, stored_embedding in embeddings.items():
#         distance = cosine(embedding, stored_embedding)
#         if distance < 0.5 and distance < min_distance:
#             min_distance = distance
#             best_match = user

#     if best_match:
#         cv2.putText(frame, f"User: {best_match}", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     else:
#         cv2.putText(frame, "Wajah baru! Tekan 's' untuk simpan", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     # Tampilkan hasil ke layar
#     cv2.imshow("Face Recognition", frame)
#     key = cv2.waitKey(30) & 0xFF

#     # Simpan wajah baru jika ditekan 's'
#     if key == ord('s') and (best_match is None or min_distance > 0.5):
#         user_name = get_username()
#         if user_name:
#             embeddings[user_name] = embedding
#             save_embeddings(embeddings)
#             print(f"✅ Wajah {user_name} berhasil disimpan.")
#         else:
#             print("❌ Penyimpanan dibatalkan.")

#     # Keluar jika tekan 'q'
#     if key == ord('q'):
#         break

# # Bersihkan semua
# cap.release()
# cv2.destroyAllWindows()






# import cv2
# import numpy as np
# from scipy.spatial.distance import cosine
# from utils import get_embedding, load_embeddings, save_embeddings

# # Load existing embeddings
# embeddings = load_embeddings()

# # Buka kamera
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Dapatkan embedding wajah
#     embedding = get_embedding(frame)
#     if embedding is not None:
#         best_match = None
#         min_distance = float("inf")

#         # Bandingkan embedding dengan database
#         for user, stored_embedding in embeddings.items():
#             distance = cosine(embedding, stored_embedding)
#             if distance < 0.5 and distance < min_distance:
#                 min_distance = distance
#                 best_match = user

#         # Jika wajah tidak cocok dengan yang ada, berarti wajah baru
#         if best_match is None or min_distance > 0.5:
#             cv2.putText(frame, "Wajah baru! Simpan?", (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # Menunggu input dari pengguna untuk menyimpan wajah baru
#             cv2.putText(frame, "Tekan 's' untuk simpan", (50, 100),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('s'):  # Jika user tekan 's', simpan wajah baru
#                 user_name = input("Masukkan nama user baru: ")
#                 embeddings[user_name] = embedding  # Simpan embedding wajah baru

#                 # Simpan embeddings terbaru ke file
#                 save_embeddings(embeddings)
#                 print(f"✅ Wajah {user_name} berhasil disimpan.")

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
