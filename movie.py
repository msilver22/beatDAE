import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img
import os

import matplotlib.pyplot as plt

# Carica il modello DAE
model = load_model('path_to_your_dae_model.h5')

# Carica l'immagine rumorosa
noisy_image_path = 'path_to_your_noisy_image.jpg'
noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)
noisy_image = cv2.resize(noisy_image, (128, 128))
noisy_image = noisy_image.astype('float32') / 255.0
noisy_image = np.reshape(noisy_image, (1, 128, 128, 1))

# Directory per salvare i frame del video
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

# Numero di iterazioni per il denoising
iterations = 10

# Applica il modello DAE iterativamente e salva i frame
for i in range(iterations):
    denoised_image = model.predict(noisy_image)
    denoised_image = np.reshape(denoised_image, (128, 128))
    
    # Salva il frame corrente
    frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
    plt.imsave(frame_path, denoised_image, cmap='gray')
    
    # Aggiorna l'immagine rumorosa per la prossima iterazione
    noisy_image = denoised_image
    noisy_image = np.reshape(noisy_image, (1, 128, 128, 1))

# Crea il video dai frame salvati
video_path = 'denoising_process.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, 1, (128, 128), False)

for i in range(iterations):
    frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    video.write(frame)

video.release()
print(f'Video salvato in {video_path}')