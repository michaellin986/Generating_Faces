from datasets import UKTDataset
from autoencoder import AutoEncoderCNN
from PIL import Image

if __name__ == '__main__':
    data_path = './UTKFace/UTKFace/'
    dataset = UKTDataset(data_path, image_height=256, image_width=256)

    latent_size = 32
    auto_encoder = AutoEncoderCNN(image_height=256,
                                  image_width=256,
                                  latent_size=latent_size,
                                  early_stop=5,
                                  verbose='vvv')

    auto_encoder.fit(dataset, validation_split=0.15)

    latents = [auto_encoder.encoder(x) for x in dataset]
    print(latents)
    decoder = auto_encoder.get_decoder()

    output = decoder(latents[0])
    img = Image.fromarray(output, "RGB")
    img.save('output.png')
