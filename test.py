import torch
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
image = test_images[0].unsqueeze(0)
noisy_image = image + 0.5 * torch.randn(*image.shape)
noisy_image = torch.clamp(noisy_image, 0., 1.)

# Apply the model for denoising
with torch.no_grad():
    denoised_image = model(noisy_image)

# Plot the original, noisy, and denoised images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot original image
axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Plot noisy image
axes[1].imshow(noisy_image.squeeze().cpu().numpy(), cmap='gray')
axes[1].set_title('Noisy Image')
axes[1].axis('off')

# Plot denoised image
axes[2].imshow(denoised_image.squeeze().cpu().numpy(), cmap='gray')
axes[2].set_title('Denoised Image')
axes[2].axis('off')

plt.show()