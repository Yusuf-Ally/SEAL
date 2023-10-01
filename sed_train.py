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

    sed16_dev = dcase_util.datasets.TUTSoundEvents_2016_DevelopmentSet(
        data_path=dataset_storage_path
    ).initialize()
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

    files = sed16_dev.audio_files
    for f in files:
        generate_features(f)

    # for fold in config.cv_fold:
    #
    #     development_file = os.path.join(metadata_path,f'/crossvalidation/meta_fold0{fold}_development.txt')
    #     print(development_file)
    #     df = pd.read_csv(f'{metadata_path}{development_file}',sep='\t')
    #     filenames = df['path'].str.rsplit('/', 1)
    #     filenames.unique()
    #     print(filenames)


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
