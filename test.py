import torch
from torchvision.utils import save_image
from src.model import DAE
from src.data_module import MNISTDataModule

import matplotlib.pyplot as plt

# Assuming you have a VAE model class defined somewhere

# Load the saved model weights
model = DAE()
model.load_state_dict(torch.load('dae_weights.pth'))
model.eval()

# Load the MNIST dataset
data_module = MNISTDataModule()
data_module.setup()
test_loader = data_module.test_dataloader()

# Get a batch of test images
test_images, _ = next(iter(test_loader))

# Select one image from the batch
noisy_image = test_images[0].unsqueeze(0)

# Apply the model for denoising
with torch.no_grad():
    denoised_image = model(noisy_image)

# Plot both images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot noisy image
axes[0].imshow(noisy_image.squeeze().cpu().numpy(), cmap='gray')
axes[0].set_title('Noisy Image')
axes[0].axis('off')

# Plot denoised image
axes[1].imshow(denoised_image.squeeze().cpu().numpy(), cmap='gray')
axes[1].set_title('Denoised Image')
axes[1].axis('off')

plt.show()