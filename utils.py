import pandas as pd
import os
from sklearn import preprocessing
import numpy as np
import dcase_util
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

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

def preprocess_data(_X, _Y, _X_val, _Y_val, _seq_len):
    # split into sequences
    _X = split_in_seqs(_X, _seq_len)
    _Y = split_in_seqs(_Y, _seq_len)

    _X_val = split_in_seqs(_X_val, _seq_len)
    _Y_val = split_in_seqs(_Y_val, _seq_len)

    return _X, _Y, _X_val, _Y_val