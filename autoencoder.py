import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
import os

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

class CNN(nn.Module):
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

        super(CNN, self).__init__()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.c1 = nn.Conv2d(kernel_size=3,
                            in_channels=3,
                            out_channels=8,
                            stride=1,
                            padding=1)
        self.m1 = nn.MaxPool2d(kernel_size=2)
        self.d1 = nn.Dropout(p=dropout)
        self.rc1 = nn.ReLU()

        self.c2 = nn.Conv2d(kernel_size=3,
                            in_channels=8,
                            out_channels=8,
                            stride=1,
                            padding=1)

        self.m2 = nn.MaxPool2d(kernel_size=2)
        self.d2 = nn.Dropout(p=dropout)
        self.rc2 = nn.ReLU()

        self.c3 = nn.Conv2d(kernel_size=3,
                            in_channels=8,
                            out_channels=8,
                            stride=1,
                            padding=1)

        self.m3 = nn.MaxPool2d(kernel_size=2)
        self.d3 = nn.Dropout(p=dropout)
        self.rc3 = nn.ReLU()

        self.c4 = nn.Conv2d(kernel_size=3,
                            in_channels=8,
                            out_channels=8,
                            stride=1,
                            padding=1)

        self.m4 = nn.MaxPool2d(kernel_size=2)
        self.d4 = nn.Dropout(p=dropout)
        self.rc4 = nn.ReLU()

        h = int(h/16)
        w = int(w/16)
        self.fc1 = nn.Linear(h * w * 8, 128)
        self.d_fc1 = nn.Dropout(p=dropout)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(128, latent_size)
        self.to(self.device)

class EncoderCNN(CNN):
    def forward(self, x):
        x = self.c1(x)
        x = self.m1(x)
        x = self.d1(x)
        x = self.rc1(x)
        x = self.c2(x)
        x = self.m2(x)
        x = self.d2(x)
        x = self.rc2(x)
        x = self.c3(x)
        x = self.m3(x)
        x = self.d3(x)
        x = self.rc3(x)
        x = self.c4(x)
        x = self.m4(x)
        x = self.d4(x)
        x = self.rc4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.d_fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        return x.flatten()

class DecoderCNN(CNN):
    def forward(self, x):
        x = self.fc2(x)
        x = self.r1(x)
        x = self.d_fc1(x)
        x = self.fc1(x)
        x = x.reshape(8, 16, 10)
        x = self.rc4(x)
        x = self.d4(x)
        x = self.m4(x)
        x = self.c4(x)
        x = self.rc3(x)
        x = self.d3(x)
        x = self.m3(x)
        x = self.c3(x)
        x = self.rc2(x)
        x = self.d2(x)
        x = self.m2(x)
        x = self.c2(x)
        x = self.rc1(x)
        x = self.d1(x)
        x = self.m1(x)
        x = self.c1(x)
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
                 tol=1e-6):

        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.hidden_size = encoder.hidden_size
        self.encoder = EncoderCNN(in_height=image_height, in_width=image_width, latent_size=latent_size)
        self.decoder = DecoderCNN(in_height=image_height, in_width=image_width, latent_size=latent_size)

        self.batch_size = batch_size
        self.verbose = verbose
        self.max_iter = max_iter

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
        x = x.view(x.shape[0], x.shape[1], -1)
        latent = self.encoder(x)
        y = self.decoder(latent)
        return y, latent

    def fit(self, train_dataset, valid_dataset):
        '''

        :param train_dataset: torch Dataset class containing tokenized training data (int64 tensors)
        :param valid_dataset: torch Dataset class containing tokenized validation data (int64 tensors)
        :return: None. trains the encoder and decoder in place.
        '''

        self.train()

        criterion = torch.nn.MSELoss()
        criterion.to(self.device)

        optimizer_enc = torch.optim.Adam(self.encoder.parameters(),
                                         lr=self.encoder_lr,
                                         weight_decay=self.encoder.l2)

        optimizer_dec = torch.optim.Adam(self.decoder.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.decoder.l2)

        # reduce the batch-size if it's bigger than the number of samples.
        if len(valid_dataset) < self.batch_size:
            warnings.warn("Validation dataset is smaller than batch_size"
                          ". \nBatch_size: {} validation_size: {}"
                          ". \nReducing validation batch size to {}".format(
                self.batch_size, len(valid_dataset), len(valid_dataset)
            ))
            valid_batch_size = len(valid_dataset)
        else:
            valid_batch_size = self.batch_size

        if len(train_dataset) < self.batch_size:
            warnings.warn("Training dataset is smaller than batch_size"
                          ". \nBatch_size: {} training_size: {}"
                          ". \nReducing training batch size to {}".format(
                self.batch_size, len(train_dataset), len(train_dataset)
            ))
            train_batch_size = len(train_dataset)
        else:
            train_batch_size = self.batch_size

        training_loader = DataLoader(train_dataset,
                                     batch_size=train_batch_size,
                                     shuffle=True, drop_last=True)

        valid_loader = DataLoader(valid_dataset,
                                  batch_size=valid_batch_size,
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
            for Xbatch, ybatch in progbar(training_loader):

                Xbatch = Xbatch.to(self.device)
                ybatch = ybatch.to(self.device)

                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()

                input_length = Xbatch.shape[1]

                Xbatch = Xbatch.view(train_batch_size, input_length, -1)

                # with torch.autocast(device_type=device_str, dtype=torch.float16): # mixed precision
                outputs, enc_hidden = self(Xbatch)

                if len(outputs.shape) == 3 and len(ybatch.shape) == 2:  # need to flatten seq outputs to fit in CEloss
                    outputs = torch.flatten(outputs, start_dim=0, end_dim=1)
                    ybatch = torch.flatten(ybatch, start_dim=0, end_dim=1)

                loss = criterion(outputs, ybatch)

                loss.backward()
                optimizer_dec.step()
                optimizer_enc.step()
                tloss.append(loss.detach().cpu().item())

            vloss = []
            with torch.no_grad():
                for Xbatch, ybatch in valid_loader:
                    Xbatch = Xbatch.to(self.device)
                    ybatch = ybatch.to(self.device)

                    input_length = Xbatch.shape[1]
                    Xbatch = Xbatch.view(valid_batch_size, input_length, -1)

                    # with torch.autocast(device_type=device_str, dtype=torch.float16):
                    outputs, _ = self(Xbatch)

                    loss = criterion(outputs, ybatch)

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
                if not self.freeze_encoder_training:
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
