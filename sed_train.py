import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import dcase_util
import sed_eval
import pandas as pd
import numpy as np
import config
import scipy.signal
from model import *
from sklearn import preprocessing
from tut_dataset import TUTDataset
import pandas as pd


def get_feature_filename(audio_filename, feature_storage_path):
    """Get feature filename from audio filename. """

    return dcase_util.utils.Path(path=audio_filename).modify(
        path_base=feature_storage_path,
        filename_extension='.npy'
    )

def get_feature_matrix(audio_filename, feature_storage_path=os.path.join('data', 'features_sed')):
    """Extract acoustic features (log mel-energies) for given audio file and store them."""

    # feature_storage_path = os.path.join(feature_storage_path,audio_filename.split("datasets/")[1].rpartition('/')[0])
    new_feature_storage_path = os.path.join(feature_storage_path, audio_filename.rpartition('/')[0])
    os.makedirs(feature_storage_path, exist_ok=True)
    # print(f'audio filename is {audio_filename}')
    # print(f'new feature storage path {new_feature_storage_path}')
    feature_filename = get_feature_filename(audio_filename, new_feature_storage_path)
    print(f'feature_filename {feature_filename}')
    if os.path.exists(feature_filename):
        return np.load(feature_filename)
    else:
        audio = dcase_util.containers.AudioContainer().load(filename=audio_filename, mono=True)
        mel_extractor = dcase_util.features.MelExtractor(n_mels=config.n_mels, win_length_seconds=config.frame_width, hop_length_seconds=config.hop_width,
                                                         fs=audio.fs)
        mel_data = mel_extractor.extract(y=audio)
        np.save(feature_filename, mel_data)
        return mel_data

def get_label_filename(audio_filename, labels_storage_path):
    """Get feature filename from audio filename. """

    return dcase_util.utils.Path(path=audio_filename).modify(
        path_base=labels_storage_path,
        filename_extension='.npy'
    )

def get_labels(audio_filename, labels_storage_path=os.path.join('data', 'labels_sed')):
    """Extract labels for given audio file and store them."""
    # NB - provide just audio filename path from dataset ie. TUT-acoustic-scenes-2016-development/audio/a099_0_30.wav

    labels_storage_path = os.path.join(labels_storage_path,audio_filename.rpartition('/')[0])
    os.makedirs(labels_storage_path, exist_ok=True)
    label_filename = get_label_filename(audio_filename, labels_storage_path)
    print(f'label filename : {label_filename}')

    if os.path.exists(label_filename):
        return np.load(label_filename)
    else:
        audio = dcase_util.containers.AudioContainer().load(filename=f'data/datasets/{audio_filename}', mono=True)

        development_file = os.path.join(metadata_path,f'/meta/meta_evaluation.txt')
        df = pd.read_csv(f'{metadata_path}{development_file}',sep='\t')
        df.rename(columns={'path':'filename','event_start_time':'onset','event_end_time':'offset'}, inplace=True)
        df.drop(columns='scene_label', inplace=True)
        df = df[df['filename'] == audio_filename]
        meta = dcase_util.containers.MetaDataContainer(df.to_dict(orient='records'))
        event_roll_encoder = dcase_util.data.EventRollEncoder(
            label_list=meta.unique_event_labels,
            time_resolution=0.02
        )

        # # # Encode
        event_roll = event_roll_encoder.encode(
            metadata_container=meta,
            length_seconds=audio.duration_sec
        )
        print(f'{event_roll.label_list}\n{label_filename}')

        # initialize label array
        if len(event_roll.label_list) == 1:
            label = np.zeros((config.n_classes, event_roll.data.shape[1]))
        else:
            label = np.zeros((config.n_classes, len(event_roll.data[1])))

        # convert labels into correct label list index
        for original_index, label_name in enumerate(event_roll.label_list):
            if label_name == 'car passing by':
                label_name = 'car'
            if label_name == 'children shouting':
                label_name = 'children'
            if label_name == 'object impact':
                label_name = '(object) impact'
            if label_name == 'people speaking':
                label_name = 'people talking'
            index = config.classes.index(label_name)
            print(f'{label_name} is index position {index} in classes list')

            label[index] = event_roll.data[original_index,:]

        print(label, label.shape)
        np.save(label_filename, label)

def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs)]
        data = data.reshape((data.shape[0] // subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1], data.shape[2]))
    return data


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def load_data(_feat_folder, _lab_folder, _metadata_path,  _fold=None):

    # load development features and labels
    development_file = os.path.join(_metadata_path, f'/crossvalidation/meta_fold0{_fold}_development.txt')
    df = pd.read_csv(f'{_metadata_path}{development_file}',sep='\t')
    unique_development_files = df['path'].unique()
    print(f'Fold {_fold} training files\n{unique_development_files}\t')

    X, Y = None, None

    for dev_file in unique_development_files:
        temp_feature = get_feature_matrix(audio_filename=dev_file)
        temp_label = get_labels(audio_filename=dev_file)

        # Avoid ValueError: zero-dimensional arrays cannot be concatenated by setting first array to temp_feature
        if X is None:
            X = temp_feature
        else:
            X = np.concatenate((X, temp_feature), 1)

        if Y is None:
            Y = temp_label
        else:
            Y = np.concatenate((Y, temp_label), 1)

    # Normalize development features
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X=X)

    print(X,X.shape)
    print(Y,Y.shape)

    # load evaluation features and labels
    evaluation_file = os.path.join(_metadata_path, f'/crossvalidation/meta_fold0{_fold}_evaluation.txt')
    df = pd.read_csv(f'{_metadata_path}{evaluation_file}', sep='\t')
    unique_evaluation_files = df['path'].unique()
    print(f'Fold {_fold} evaluation files\n{unique_evaluation_files}\t')

    X_val, Y_val = None, None

    for eval_file in unique_evaluation_files:
        temp_feature = get_feature_matrix(audio_filename=eval_file)
        temp_label = get_labels(audio_filename=eval_file)

        # Avoid ValueError: zero-dimensional arrays cannot be concatenated by setting first array to temp_feature
        if X_val is None:
            X_val = temp_feature
        else:
            X_val = np.concatenate((X_val, temp_feature), 1)

        if Y_val is None:
            Y_val = temp_label
        else:
            Y_val = np.concatenate((Y_val, temp_label), 1)

    # Normalize evaluation features
    X_val = scaler.fit_transform(X=X_val)

    print(X_val,X_val.shape)
    print(Y_val,Y_val.shape)

    return X, Y, X_val, Y_val



def preprocess_data(_X, _Y, _X_val, _Y_val, _seq_len):
    # split into sequences
    _X = split_in_seqs(_X, _seq_len)
    _Y = split_in_seqs(_Y, _seq_len)

    _X_val = split_in_seqs(_X_val, _seq_len)
    _Y_val = split_in_seqs(_Y_val, _seq_len)

    return _X, _Y, _X_val, _Y_val



def train():
    #     Per fold:
    # 1. Generate X, X_eval and Y, Y_eval data
    # 2. Train model using MSE
    # 3. After training, evaluate on eval data using SED metrics
    # 4. After training on all folds and epochs for each fold, test on test metadata

    # Arguments & parameters
    stop_iteration = 150
    learning_rate = 1e-3
    patience = int(0.6*stop_iteration)
    holdout_fold = np.arange(1, 4)
    seq_len = 500
    batch_size = 32

    # initialization
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

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
    model_filename = os.path.join(data_storage_path, 'SED_model.h5')

    avg_f1 = []
    avg_error = []
    print(f'Learning rate {config.learning_rate} - sequence length {config.sequence_length} - batch_size {config.batch_size}')


    # For evaluating the model
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=config.classes,
        time_resolution=1.0
    )

    # Get all audiofiles
    # for fold in config.cv_fold:
    #     if fold == 1:
    #         development_file = os.path.join(metadata_path,f'/crossvalidation/meta_fold0{fold}_development.txt')
    #         # print(development_file)
    #         df = pd.read_csv(f'{metadata_path}{development_file}',sep='\t')
    #         # filenames = df['path'].str.rsplit('/', 1)
    #         files = df['path'].unique()
    #         # print(un)
    #         print(files)

    # Load features and labels
    for fold in config.cv_fold:

        if fold==5:
            # Load features and labels
            X, Y, X_val, Y_val = load_data(_feat_folder=feature_storage_path,_lab_folder=labels_storage_path, _metadata_path=metadata_path,_fold=fold)

            X = np.transpose(X)
            X_val = np.transpose(X_val)
            Y = np.transpose(Y)
            Y_val = np.transpose(Y_val)

            # Stack data into clips of desired sequence length
            X, Y, X_val, Y_val = preprocess_data(X, Y, X_val, Y_val, config.sequence_length)
            np.save('/Users/yusuf/Downloads/fold1_Xval.npy',X_val)
            np.save('/Users/yusuf/Downloads/fold1_Yval.npy', Y_val)

        if fold ==1:
            X = np.load('/Users/yusuf/Downloads/fold1_X.npy')
            X_val = np.load('/Users/yusuf/Downloads/fold1_Xval.npy')
            Y = np.load('/Users/yusuf/Downloads/fold1_Y.npy')
            Y_val = np.load('/Users/yusuf/Downloads/fold1_Yval.npy')




        train_dataset = TUTDataset(X, Y)
        validate_dataset = TUTDataset(X_val, Y_val)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                                                   num_workers=1, pin_memory=True)

        validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=config.batch_size, shuffle=True,
                                                      num_workers=1, pin_memory=True)



        # Prepare model
        modelcrnn = CRNN(config.n_classes, config.cnn_filter, config.gru_hidden_layers, config.dropout)

        if 'cuda' in device:
            modelcrnn.to(device)
        print('\nCreate model:')

        # Optimizer
        # optimizer = optim.Adam(CRNN.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=False)
        optimizer = optim.Adam(modelcrnn.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=False)

        best_epoch = 0; pat_cnt = 0; pat_learn_rate = 0; best_loss = 99999
        tr_loss, val_F1, val_ER = [0] * stop_iteration, [0] * stop_iteration, [0] * stop_iteration

        # Train on mini batches
        tr_batch_loss = list()
        for epoch in range(stop_iteration):

            modelcrnn.train()
            # TRAIN
            for (batch_data, batch_target) in train_loader:
                # Zero gradients for every batch
                optimizer.zero_grad()

                batch_output = modelcrnn(move_data_to_device(batch_data, device))

                # Calculate loss
                loss = clip_mse(batch_output, move_data_to_device(batch_target, device))

                tr_batch_loss.append(loss.item())

                # Backpropagation
                loss.backward()
                optimizer.step()

            tr_loss[epoch] = np.mean(tr_batch_loss)
            print(f'Epoch {epoch} loss: {tr_loss[epoch]}\n')
        #
        #     # VALIDATE
        #     modelcrnn.eval()
        #
        #     with torch.no_grad():
        #
        #         segment_based_metrics_batch = sed_eval.sound_event.SegmentBasedMetrics(
        #             event_label_list=config.labels_hard,
        #             time_resolution=1.0
        #         )
        #
        #         running_loss = 0.0
        #         for (batch_data, batch_target) in validate_loader:
        #             batch_output = modelcrnn(move_data_to_device(batch_data, device))
        #
        #             loss = clip_mse(batch_output, move_data_to_device(batch_target, device))
        #
        #             segment_based_metrics_batch = metric_perbatch(segment_based_metrics_batch,
        #                                                           batch_output.reshape(-1,
        #                                                                                len(config.labels_soft)).detach().cpu().numpy(),
        #                                                           batch_target.reshape(-1,
        #                                                                                len(config.labels_soft)).numpy())
        #
        #             running_loss += loss
        #
        #         avg_vloss = running_loss / len(validate_loader)
        #
        #         batch_segment_based_metrics_ER = segment_based_metrics_batch.overall_error_rate()
        #         batch_segment_based_metrics_f1 = segment_based_metrics_batch.overall_f_measure()
        #         val_F1[epoch] = batch_segment_based_metrics_f1['f_measure']
        #         val_ER[epoch] = batch_segment_based_metrics_ER['error_rate']
        #
        #         # Check if during the epochs the ER does not improve
        #         if avg_vloss < best_loss:
        #             best_model = modelcrnn
        #             best_epoch = epoch
        #             best_loss = avg_vloss
        #             pat_cnt = 0
        #             pat_learn_rate = 0
        #             output = segment_based_metrics_batch.result_report_class_wise()
        #
        #             print(output)
        #             torch.save(best_model.state_dict(), f'{output_model}/best_fold{fold}.bin')
        #
        #     pat_cnt += 1
        #     pat_learn_rate += 1
        #
        #     if pat_learn_rate > int(0.3 * stop_iteration):
        #         for g in optimizer.param_groups:
        #             g['lr'] = g['lr'] / 10
        #             pat_learn_rate = 0
        #             print(f'\tDecreasing learning rate to:{g["lr"]}')
        #
        #     print(f'Epoch: {epoch} - Train loss: {round(tr_loss[epoch], 3)} - Val loss: {round(avg_vloss.item(), 3)}'
        #           f' - val F1 {round(val_F1[epoch] * 100, 2)} - val ER {round(val_ER[epoch], 3)}'
        #           f' - best epoch {best_epoch} F1 {round(val_F1[best_epoch] * 100, 2)}')
        #
        #     segment_based_metrics_batch.reset()
        #     # Stop learning
        #     if (epoch == stop_iteration) or (pat_cnt > patience):
        #         break





if __name__ == '__main__':
    # train()

    # Paths to store data
    data_storage_path = 'data'
    dataset_storage_path = os.path.join(data_storage_path, 'datasets')
    feature_storage_path = os.path.join(data_storage_path, 'features_sed')
    dcase_util.utils.Path().create(
        [data_storage_path, dataset_storage_path, feature_storage_path]
    )

    train()

    # t = get_feature_matrix('TUT-sound-events-2016-development/audio/residential_area/b003.wav',feature_storage_path=feature_storage_path)
    # print(t)