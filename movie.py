import matplotlib.pyplot as plt
import os
import imageio

def animate_transition(self, noisy_img, denoised_img, idx):
        """ Crea un'animazione per mostrare la transizione da immagine corrotta a denoised """
        noisy_img = noisy_img.cpu().detach().numpy()
        denoised_img = denoised_img.cpu().detach().numpy()
        
        # Crea una directory temporanea per salvare i frame
        os.makedirs('reconstructions/animation', exist_ok=True)
        images = []

        # Genera frame interpolando tra noisy e denoised
        for alpha in np.linspace(0, 1, 20):
            interpolated = (1 - alpha) * noisy_img + alpha * denoised_img
            
            # Salva ogni frame come immagine
            fig, ax = plt.subplots()
            ax.imshow(interpolated[0], cmap='gray')
            ax.axis('off')
            filename = f'reconstructions/animation/frame_{idx}_{int(alpha*100):03d}.png'
            plt.savefig(filename)
            plt.close()
            images.append(imageio.imread(filename))

        # Crea la GIF animata per l'immagine corrente
        imageio.mimsave(f'reconstructions/animation/transition_{idx}.gif', images, fps=10)
        print(f"[LOG] GIF salvata: reconstructions/animation/transition_{idx}.gif")