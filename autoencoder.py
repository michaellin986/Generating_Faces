import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
import os
from copy import deepcopy

warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm.auto import tqdm

tqdm.pandas()

class EarlyStopper:
    '''
    class to help with early stopping criteria
    '''

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EncoderCNN(nn.Module):
    """
    For input [3 x 128 x 160]

    -----
    conv2d: kernel size 3, 16 out channels, padding 1
    [16 x 128 x 160]
    maxpool: kernel size 2
    [16 x 64 x 80]
    ReLU
    [16 x 64 x 80]
    ------

    ------
    conv2d: kernel size 3, 16 our channels, padding 1
    [16 x 32 x 40]
    maxpool: kernel size 2
    [16 x 32 x 40]
    ReLU
    [16 x 32 x 40]
    ------

    ------
    conv2d: kernel size 3, 16 our channels, padding 1
    [16 x 32 x 40]
    maxpool: kernel size 2
    [16 x 16 x 20]
    ReLU
    [16 x 16 x 20]
    ------

    ------
    conv2d: kernel size 3, 16 our channels, padding 1
    [16 x 16 x 20]
    maxpool: kernel size 2
    [16 x 8 x 10]
    ReLU
    [16 x 8 x 10]
    ------

    linear: in (8 * 10 * 16): out ( 10 * 16 )
    dropout
    linear: in ( 10 * 16): out ( 10 * 16 )
    dropout
    linear: in ( 10 * 16): out [num_actions]
    [num_actions]
    """

    def __init__(self,
                 in_height: int,
                 in_width: int,
                 dropout: float = 0.1,
                 latent_size: int = 128):

        h = in_height
        w = in_width

        super().__init__()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layer1 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=1, out_channels=128,
                            stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU()
                )

        self.layer3 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

        self.layer4 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                )

        self.layer5 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU()
                )

        self.layer6 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

        self.layer7 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU()
                )

        self.layer8 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU()
                )

        self.layer9 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )

        self.layer10 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU()
                )

        self.layer11 = nn.Sequential(
                    nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128,
                                    stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.Dropout(p=dropout),
                    nn.ReLU()
                )

        self.fc1 = nn.Linear(h * w * 2, 512)
        self.d_fc1 = nn.Dropout(p=dropout)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(512, latent_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x, use_cpu=False):
        if use_cpu:
            x = x.to(self.device)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.d_fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        if use_cpu:
            x = x.cpu() # return to cpu
        return x

class DecoderCNN(nn.Module):

    def __init__(self,
                 in_height: int,
                 in_width: int,
                 dropout: float = 0.1,
                 latent_size: int = 128):

        h = in_height
        w = in_width

        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(kernel_size=3, 
                      in_channels=128,
                      out_channels=1,
                      stride=1, padding=1),
            nn.BatchNorm2d(1),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer9 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(kernel_size=3, 
                      in_channels=128, 
                      out_channels=128,
                      stride=1, padding=1),
            nn.BatchNorm2d(128),                     
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

        h = int(h)
        w = int(w)
        self.h = h
        self.w = w
        self.fc1 = nn.Linear(512, h * w * 2)
        self.d_fc1 = nn.Dropout(p=dropout)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(latent_size, 512)
        self.r2 = nn.ReLU()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x, use_cpu=False):
        if use_cpu:
            x = x.to(self.device)

        x = self.fc2(x)
        x = self.r2(x)
        x = self.d_fc1(x)

        x = self.fc1(x)
        x = self.r1(x)
        x = self.d_fc1(x)
        x = x.view(x.shape[0], 128, int(self.h/8), int(self.w/8))

        x = self.layer11(x)
        x = self.layer10(x)
        x = self.layer9(x)
        x = self.layer8(x)
        x = self.layer7(x)
        x = self.layer6(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        if use_cpu:
            x = x.cpu() # return to cpu
        return x


class AutoEncoderCNN(nn.Module):

    def __init__(self, image_height: int = 512,
                image_width: int = 512,
                latent_size: int = 128,
                 lr=1e-4,
                 batch_size=32,
                 verbose='vv',
                 max_iter=200,
                 early_stop=5,
                 tol=1e-6,
                 use_learning_schedule=True):

        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.hidden_size = latent_size
        self.encoder = EncoderCNN(in_height=image_height, in_width=image_width, latent_size=latent_size)
        self.decoder = DecoderCNN(in_height=image_height, in_width=image_width, latent_size=latent_size)

        self.batch_size = batch_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.use_learning_schedule = use_learning_schedule

        self.lr = lr
        self.tol = tol
        self.early_stop = early_stop
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self, x):
        """

        :param x: input features to the encoder model
        :return: output of the decoder
        """
        x = x.transpose(dim0=3, dim1=1) # torch needs channel at dim1. We have at dim3
        latent = self.encoder(x)
        y = self.decoder(latent)
        y = y.transpose(dim0=3, dim1=1) # switch dim1 and dim3 back
        return y, latent

    def fit(self, dataset, validation_split: float = 0.15):
        '''

        :param dataset: torch Dataset class containing tokenized training data (int64 tensors)
        :return: None. trains the encoder and decoder in place.
        '''

        self.train()

        criterion = torch.nn.MSELoss()
        criterion.to(self.device)

        optimizer_enc = torch.optim.Adam(self.encoder.parameters(),
                                         lr=self.lr)

        optimizer_dec = torch.optim.Adam(self.decoder.parameters(),
                                         lr=self.lr)

        valid_dataset = deepcopy(dataset)
        valid_dataset.X = dataset.X[int(validation_split * len(dataset)):]
        train_dataset = dataset
        train_dataset.X = train_dataset.X[:int((1 - validation_split) * len(dataset))]
        print('train size: ', len(train_dataset))

        training_loader = DataLoader(train_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True, drop_last=True)

        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True, drop_last=True)

        enc_scheduler = MultiStepLR(optimizer_enc, milestones=[100, 150], gamma=0.25)
        dec_scheduler = MultiStepLR(optimizer_dec, milestones=[100, 150], gamma=0.25)

        early_stopper = EarlyStopper(patience=self.early_stop, min_delta=0)

        reason = "Max iterations reached"  # ititialise a reason for exitting training

        learning_curve = []
        validation_curve = []
        ave_valid_loss = 1e6  # initialise to something high
        ave_train_loss = 1e6

        if self.verbose == 'vvv':
            progbar = tqdm
        else:
            progbar = lambda x: x

        for epoch in range(self.max_iter):
            tloss = []
            for Xbatch in progbar(training_loader):

                Xbatch = Xbatch.to(self.device)

                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()

                # with torch.autocast(device_type=device_str, dtype=torch.float16): # mixed precision
                outputs, enc_hidden = self(Xbatch)

                loss = criterion(outputs, Xbatch)

                loss.backward()
                optimizer_dec.step()
                optimizer_enc.step()
                tloss.append(loss.detach().cpu().item())

            vloss = []
            with torch.no_grad():
                for Xbatch in valid_loader:
                    Xbatch = Xbatch.to(self.device)

                    # with torch.autocast(device_type=device_str, dtype=torch.float16):
                    outputs, _ = self(Xbatch)

                    loss = criterion(outputs, Xbatch)

                    vloss.append(loss.detach().cpu().item())

            ave_train_loss = sum(tloss) / len(tloss)
            ave_valid_loss = sum(vloss) / len(vloss)

            learning_curve.append(ave_train_loss)
            validation_curve.append(ave_valid_loss)

            # stopping criteria
            if self.tol is not None:
                if ave_train_loss <= self.tol:
                    reason = "\nLearning criteria achieved"
                    break
            if early_stopper.early_stop(ave_valid_loss):
                reason = "\nEarly stopping achieved"
                break

            # print statistics
            if epoch % 1 == 0:  # print out loss as model is training
                if self.verbose == 'vv':
                    print('\r[%d] train loss: %.6f valid loss: %.6f' %
                          (epoch + 1, ave_train_loss, ave_valid_loss), end='', flush=True)
                if self.verbose == 'vvv':
                    print('[%d] train loss: %.6f valid loss: %.6f' %
                          (epoch + 1, ave_train_loss, ave_valid_loss))

            if self.use_learning_schedule:
                enc_scheduler.step()
                dec_scheduler.step()

        if self.verbose == 'v' or self.verbose == 'vv':
            print("\n...Training complete: %s \n final loss (train/valid) %.5f/%.5f" % (
                reason, ave_train_loss, ave_valid_loss))

        self.eval()
        return learning_curve, validation_curve

    def save_model(self, modelname):
        torch.save(self.encoder.state_dict(), os.path.join('./TorchModels/', 'enc_' + modelname))
        torch.save(self.decoder.state_dict(), os.path.join('./TorchModels/', 'dec_' + modelname))

    def load_model(self, modelname):
        enc = torch.load(os.path.join('./TorchModels/', 'enc_' + modelname),
                         map_location=self.device)
        self.encoder.load_state_dict(enc, strict=False)
        dec = torch.load(os.path.join('./TorchModels/', 'dec_' + modelname),
                         map_location=self.device)
        self.decoder.load_state_dict(dec, strict=False)

        self.encoder.eval()
        self.decoder.eval()
