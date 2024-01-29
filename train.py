import utils
from model import CRNN
from tut_dataset import TUTDataset
import os
import numpy as np
import config
import dcase_util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
def train(model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)
def lr_scheduler(optimizer, epoch, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

def main():

    # Paths to store data
    data_storage_path = 'data'
    dataset_storage_path = os.path.join(data_storage_path, 'datasets')
    feature_storage_path = os.path.join(data_storage_path, 'features_sed')
    metadata_path = os.path.join(data_storage_path, 'metadata')
    labels_storage_path = os.path.join(data_storage_path, ' labels_sed')
    dcase_util.utils.Path().create(
        [data_storage_path, dataset_storage_path, feature_storage_path]
    )

    # Filename for acoustic model
    # model_filename = os.path.join(data_storage_path, 'SED_model.h5')

    # Initialize the model, criterion, and optimizer | compare against initializing outside fold loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(config.n_classes, config.cnn_filter, config.gru_hidden_layers, config.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Initialize the TensorBoard summary writer
    writer = SummaryWriter()

    # Initialize the best validation loss and early stopping counter
    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Train the model for each fold
    for fold in config.cv_fold:
        
        # # Create the dataloaders for train and validation phases
        # X, Y, X_val, Y_val = utils.load_data(_feat_folder=feature_storage_path, _lab_folder=labels_storage_path,
        #                                _metadata_path=metadata_path, _fold=fold)
        #
        # X = np.transpose(X)
        # X_val = np.transpose(X_val)
        # Y = np.transpose(Y)
        # Y_val = np.transpose(Y_val)
        #
        # # Stack data into clips of desired sequence length
        # X, Y, X_val, Y_val = utils.preprocess_data(X, Y, X_val, Y_val, config.sequence_length)

        if fold == 1:  # temp - remove
            X = np.load('/Users/yusuf/Downloads/fold1_X.npy')[:50]
            X_val = np.load('/Users/yusuf/Downloads/fold1_Xval.npy')[:15]
            Y = np.load('/Users/yusuf/Downloads/fold1_Y.npy')[:50]
            Y_val = np.load('/Users/yusuf/Downloads/fold1_Yval.npy')[:15]

        train_dataset = TUTDataset(X, Y)
        val_dataset = TUTDataset(X_val, Y_val)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1)

        # Initialize the model, criterion, and optimizer | compare against initializing once outside fold loop
        model = CRNN(config.n_classes, config.cnn_filter, config.gru_hidden_layers, config.dropout).to(device)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Initialize the training and validation losses
        train_losses = []
        val_losses = []

        # Train the model for the current fold
        for epoch in range(num_epochs):

            # Train the model for the current epoch
            train_loss = train(model, criterion, optimizer, train_dataloader, device)
            train_losses.append(train_loss.item())

            # Validate the model for the current epoch
            val_loss = validate(model, criterion, val_dataloader, device)
            val_losses.append(val_loss.item())

            # Update the learning rate scheduler
            lr_scheduler(optimizer, epoch, lr_decay)

            # Log the training and validation losses to TensorBoard
            writer.add_scalar('fold_{}/train_loss'.format(fold), train_loss, epoch)
            writer.add_scalar('fold_{}/val_loss'.format(fold), val_loss, epoch)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print('Early stopping at epoch', epoch)
                    break

        # Save the best model for the current fold
        torch.save(model.state_dict(), 'best_model_fold_{}.pt'.format(fold))

    # Close the TensorBoard summary writer
    writer.close()


if __name__=="__main__":
    main()