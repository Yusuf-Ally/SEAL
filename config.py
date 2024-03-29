import numpy as np

n_classes = 25
sequence_length = 500
sampling_freq = 44100
n_fft = 2048
batch_size = 32
n_mels = 64
cnn_filter = 128
gru_hidden_layers = 32
dropout = 0.3
frame_width = 0.04
hop_width = 0.02

# Arguments & parameters
epochs = 100
bce_alpha = 1
bce_beta=1
stop_iteration = 100
learning_rate = 0.01
learning_rate_decay = 0.1
# patience = int(0.6*stop_iteration)
early_stopping_patience = 15
cv_fold = np.arange(1, 5)

classes = ['(object) banging','(object) impact','(object) rustling','(object) snapping','(object) squeaking','bird singing','brakes squeaking','breathing','car','children','cupboard','cutlery','dishes','drawer','fan','glass jingling','keyboard typing','large vehicle','mouse clicking','mouse wheeling','people talking','people walking','washing dishes','water tap running','wind blowing']

