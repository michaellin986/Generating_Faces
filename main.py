from datasets import UKTDataset
from autoencoder import AutoEncoderCNN


if __name__ == '__main__':

    datapath = './archive/UTKFace/'
    dataset = UKTDataset(datapath, image_height=512, image_width=512)
    
    latent_size = 128
    autoencoder = AutoEncoderCNN(image_height=512, image_width=512, latent_size=latent_size, verbose='vvv')

    autoencoder.fit(dataset, dataset)
