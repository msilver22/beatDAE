import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
import torch
from src.model import DAE
from src.data_module import MNISTDataModule
import numpy as np
from matplotlib.animation import FuncAnimation

def animate_transition(noisy_img, denoised_img):

    noisy_img = noisy_img.cpu().detach().numpy()
    denoised_img = denoised_img.cpu().detach().numpy()
    
    # Crea una directory temporanea per salvare i frame
    os.makedirs('animation', exist_ok=True)
    images = []

    fig, ax = plt.subplots()
    def update(alpha):
        interpolated = (1 - alpha) * noisy_img + alpha * denoised_img
        ax.imshow(interpolated.squeeze(), cmap='gray')
        ax.axis('off')
        return ax

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 20), repeat=False)
    plt.show()

    for alpha in np.linspace(0, 1, 20):
        interpolated = (1 - alpha) * noisy_img + alpha * denoised_img
        
        # Salva ogni frame come immagine
        fig, ax = plt.subplots()
        ax.imshow(interpolated.squeeze(), cmap='gray')
        ax.axis('off')
        filename = f'animation/frame_{int(alpha*100):03d}.png'
        plt.savefig(filename)
        plt.close()
        images.append(imageio.imread(filename))

    imageio.mimsave(f'animation/transition.gif', images, fps=10)
    print(f"[LOG] GIF salvata: animation/transition.gif")

model = DAE()
model.load_state_dict(torch.load('dae_weights.pth', weights_only=True))
model.eval()

data_module = MNISTDataModule()
data_module.setup()
test_loader = data_module.test_dataloader()

test_images, _ = next(iter(test_loader))
image = test_images[2].unsqueeze(0)
noisy_image = image + 0.5 * torch.randn(*image.shape)
noisy_image = torch.clamp(noisy_image, 0., 1.)
with torch.no_grad():
    denoised_image = model(noisy_image)

animate_transition(noisy_image, denoised_image)