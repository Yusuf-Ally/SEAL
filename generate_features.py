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

    feature_storage_path = os.path.join(feature_storage_path,audio_filename.split("datasets/")[1].rpartition('/')[0])
    os.makedirs(feature_storage_path, exist_ok=True)
    print(f'feature storage path {feature_storage_path}')
    feature_filename = get_feature_filename(audio_filename, feature_storage_path)
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

def generate_features(audio_filename):

    features = get_feature_matrix(audio_filename, feature_storage_path)
    # print(f'{audio_filename}, {features.shape}')

def get_label_filename(audio_filename, labels_storage_path):
    """Get feature filename from audio filename. """

    return dcase_util.utils.Path(path=audio_filename).modify(
        path_base=labels_storage_path,
        filename_extension='.npy'
    )

def get_labels(audio_filename, labels_storage_path=os.path.join('data', 'labels_sed')):
    """Extract labels (log mel-energies) for given audio file and store them."""
    # NB - provide just audio filename path from dataset ie. TUT-acoustic-scenes-2016-development/audio/a099_0_30.wav

    # long_filename = audio_filename
    # audio_filename = audio_filename.split('data/')[1]

    labels_storage_path = os.path.join(labels_storage_path,audio_filename.rpartition('/')[0])
    os.makedirs(labels_storage_path, exist_ok=True)
    # print(f'labels storage path {labels_storage_path}')
    label_filename = get_label_filename(audio_filename, labels_storage_path)
    # print(f'label_filename {label_filename}')
    if os.path.exists(label_filename):
        return np.load(label_filename)
    else:
        audio = dcase_util.containers.AudioContainer().load(filename=f'data/datasets/{audio_filename}', mono=True)
        # mel_extractor = dcase_util.features.MelExtractor(n_mels=config.n_mels, win_length_seconds=config.frame_width, hop_length_seconds=config.hop_width,
        #                                                  fs=audio.fs)

        development_file = os.path.join(metadata_path,f'/meta/meta_evaluation.txt')
        df = pd.read_csv(f'{metadata_path}{development_file}',sep='\t')
        df.rename(columns={'path':'filename','event_start_time':'onset','event_end_time':'offset'}, inplace=True)
        df.drop(columns='scene_label', inplace=True)
        df = df[df['filename'] == audio_filename]
        # print(df)
        meta = dcase_util.containers.MetaDataContainer(df.to_dict(orient='records'))
        #
        event_roll_encoder = dcase_util.data.EventRollEncoder(
            label_list=meta.unique_event_labels,
            time_resolution=0.02
        )
        # #
        # # # Encode
        event_roll = event_roll_encoder.encode(
            metadata_container=meta,
            length_seconds=audio.duration_sec
        )

        # print(event_roll.data.shape[1])
        # print(event_roll.data)
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


        # event_roll = meta.to_event_roll(
        #     label_list=config.classes,  # Event labels
        #     time_resolution=config.hop_width,  # Time resolution of feature matrix
        #     length_seconds=audio.duration_sec  # Length of original audio signal
        # )
        # target_matrix = data_sequencer.sequence(event_roll)
        # print(target_matrix,target_matrix.shape)


        # np.save(label_filename,event_roll.data)
        # print(f'Generated label {label_filename}')
        # event_roll.plot()

        # # Visualize
        # event_roll.plot()
        # np.save(label_filename, event_roll_encoder.save())
        # return label_data

def train():

    # 1. Generate X and Y data
    # 2. Per fold, train model using MSE
    # 3. After training, evaluate on eval data using SED metrics
    # 4. After training on all folds and epochs for each fold, test on test metadata

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

    # For evaluating the model
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=config.classes,
        time_resolution=1.0
    )

    # sed16_dev = dcase_util.datasets.TUTSoundEvents_2016_DevelopmentSet(
    #     data_path=dataset_storage_path
    # ).initialize()
    # sed16_eval = dcase_util.datasets.TUTSoundEvents_2016_EvaluationSet(
    #     data_path=dataset_storage_path
    # ).initialize()
    #
    # sed17_dev = dcase_util.datasets.TUTSoundEvents_2017_DevelopmentSet(
    #     data_path=dataset_storage_path
    # ).initialize()
    # sed17_eval = dcase_util.datasets.TUTSoundEvents_2017_EvaluationSet(
    #     data_path=dataset_storage_path
    # ).initialize()
    #
    # acsc16_dev = dcase_util.datasets.TUTAcousticScenes_2016_DevelopmentSet(
    #     data_path=dataset_storage_path
    # ).initialize()
    # acsc16_eval = dcase_util.datasets.TUTAcousticScenes_2016_EvaluationSet(
    #     data_path=dataset_storage_path
    # ).initialize()
    #
    # acsc17_dev = dcase_util.datasets.TUTAcousticScenes_2017_DevelopmentSet(
    #     data_path=dataset_storage_path
    # ).initialize()
    # acsc17_eval = dcase_util.datasets.TUTAcousticScenes_2017_EvaluationSet(
    #     data_path=dataset_storage_path
    # ).initialize()

    # files = acsc16_dev.audio_files
    # for f in files:
    #     # generate_features(f)
    #     get_labels(f)
    #     # print(f)

    # dir1 = 'data/datasets/TUT-sound-events-2017-evaluation/audio/street/'
    # for dirpath, dirnames, filenames in os.walk(dir1):
    #     for filename in filenames:
    #         generate_features(os.path.join(dir1,filename))

    # for fold in config.cv_fold:
    #     if fold == 1:
    #         development_file = os.path.join(metadata_path,f'/crossvalidation/meta_fold0{fold}_development.txt')
    #         # print(development_file)
    #         df = pd.read_csv(f'{metadata_path}{development_file}',sep='\t')
    #         files = df['path'].unique()
    #         df.rename(columns={'path':'filename','event_start_time':'onset','event_end_time':'offset'}, inplace=True)
    #         print(df.columns)
    #         df.drop(columns='scene_label', inplace=True)




if __name__ == '__main__':
    # train()

    # Paths to store data
    data_storage_path = 'data'
    dataset_storage_path = os.path.join(data_storage_path, 'datasets')
    feature_storage_path = os.path.join(data_storage_path, 'features_sed')
    metadata_storage_path = os.path.join(data_storage_path, 'features_sed')
    metadata_path = os.path.join(data_storage_path, 'metadata')
    dcase_util.utils.Path().create(
        [data_storage_path, dataset_storage_path, feature_storage_path]
    )

    # development_file = os.path.join(metadata_path, f'/meta/meta_evaluation.txt')
    # df = pd.read_csv(f'{metadata_path}{development_file}', sep='\t')
    # df['duration'] = df.event_end_time - df.event_start_time
    # print(df['duration'].sum())
    # train()
    dataset_dev = dcase_util.datasets.Dataset(name='TUT_SED_ASC_Combined',data_path=dataset_storage_path,local_path=data_storage_path, meta_filename='metadata/meta/meta_development_db.txt',crossvalidation_folds=4,evaluation_setup_folder= 'metadata/crossvalidation/')
    dataset_eval = dcase_util.datasets.Dataset(name='TUT_SED_ASC_Combined', data_path=dataset_storage_path,
                                              local_path=data_storage_path,
                                              meta_filename='metadata/meta/meta_evaluation_db.txt',
                                              crossvalidation_folds=4,
                                              evaluation_setup_folder='metadata/crossvalidation/')
    #
    dev_files = []
    eval_files = []

    for i,file in enumerate(dataset_eval.train_files()):
        filename = file.split('data/')[1]
        print(i,filename)
        # get_labels(filename)
        eval_files.append(filename)


    with open('eval_files.txt','w') as tfile:
        tfile.write('\n'.join(eval_files))

