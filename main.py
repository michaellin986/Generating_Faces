from datasets import UKTDataset
from autoencoder import AutoEncoderCNN


if __name__ == '__main__':

    datapath = './archive/UTKFace/'
    dataset = UKTDataset(datapath, image_height=256, image_width=256)
    
    latent_size = 128
    autoencoder = AutoEncoderCNN(image_height=256, image_width=256, latent_size=latent_size, verbose='vvv')

    autoencoder.fit(dataset, dataset)
