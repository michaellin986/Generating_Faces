#%%
from datasets import UKTDataset
from autoencoder import AutoEncoderCNN
from PIL import Image
import numpy as np

if __name__ == '__main__':
    data_path = './UTKFace/UTKFace/'
    dataset = UKTDataset(data_path, image_height=128, image_width=128)

    latent_size = 256
    auto_encoder = AutoEncoderCNN(image_height=128,
                                  image_width=128,
                                  latent_size=latent_size,
                                  early_stop=10,
                                  batch_size=16,
                                  lr=1e-3,
                                  max_iter=1000,
                                  verbose='vvv')

    auto_encoder.fit(dataset, validation_split=0.15)
    
    encoder = auto_encoder.get_encoder()
    decoder = auto_encoder.get_decoder()

    latents = [encoder(x.unsqueeze(0).transpose(dim0=1, dim1=3), use_cpu=True) for x in dataset[0:10]]

    for i in range(len(latents)):
        imgin = Image.fromarray(np.uint8(255*dataset[i].squeeze(-1).numpy()), 'L')
        imgin.save('./results/input{}.png'.format(i))
        output = decoder(latents[i], use_cpu=True).squeeze(0).squeeze(0)
        imgout = Image.fromarray(np.uint8(255*output.detach().numpy()).transpose(), 'L')
        imgout.save('./results/output{}.png'.format(i))

# %%
