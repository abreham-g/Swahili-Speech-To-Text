import streamlit as st
import wavio
import librosa
import sounddevice as sd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import * 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras import backend as K

def character_dict():
    alphabet = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
    supported = alphabet.split()

    char_map = {}
    char_map[""] = 0
    char_map["<SPACE>"] = 1
    idx = 2
    for c in supported:
        char_map[c] = idx
        idx += 1
    index_map = {v: k for k, v in char_map.items()}
    return char_map, index_map

char_map, index_map = character_dict()

def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        elif c in alphabets:
            ch = char_map[c]
        else:
            print(c)
            print('character not found')
            break
        int_sequence.append(ch)
    return np.array(int_sequence)

def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    textch = []
    for c in int_sequence:
        ch = index_map[c]
        textch.append(ch)
    text = ''.join(textch)
    text = text.replace('<SPACE>', ' ')
    return text

class LogMelSpectrogram(tf.keras.layers.Layer):
    """Compute log-magnitude mel-scaled spectrograms."""

    def __init__(self, sample_rate, fft_size, hop_size, n_mels,
                 f_min=0.0, f_max=None, **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
            The corresponding batch of log-mel-spectrograms
        """
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def power_to_db(magnitude, amin=1e-16, top_db=80.0):
            """
            https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
            """
            ref_value = tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

            return log_spec

        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      pad_end=False)

        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                     self.mel_filterbank)

        log_mel_spectrograms = power_to_db(mel_spectrograms)

        # add channel dimension
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 3)

        return log_mel_spectrograms

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'n_mels': self.n_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(LogMelSpectrogram, self).get_config())

        return config

def preprocessin_model(sample_rate, fft_size, frame_step, n_mels, mfcc=False):

    input_data = Input(name='input', shape=(None,), dtype="float32")
    featLayer = LogMelSpectrogram(
        fft_size=fft_size,
        hop_size=frame_step,
        n_mels=n_mels,
        
        sample_rate=sample_rate,
        f_min=0.0,
        
        f_max=int(sample_rate / 2)
    )(input_data)
    
    x = BatchNormalization()(featLayer)
    model = Model(inputs=input_data, outputs=x, name="preprocessin_model")

    return model

def conv_rnn(n_mels, output_dim=224, rnn_layers=4, units=400, drop_out=0.5, act='tanh'):

    input_data = Input(name='the_input', shape=(None, n_mels, 1))

    y = Conv2D(32, (3, 3), padding='same')(input_data)  # was 32
    y = Activation('relu')(y)
    y = BatchNormalization()(y)

    x = MaxPooling2D((1, 2))(y)

    x = Conv2D(64, (3, 3), padding='same')(y)  # was 32
    x = Activation('relu')(x)
    y = BatchNormalization()(y)

    x = MaxPooling2D((1, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    y = BatchNormalization()(y)

    x = MaxPooling2D((1, 2))(x)

    x = Dense(128)(x)
    x = Dense(64)(x)
    x = Dense(32)(x)

    x = Reshape((-1, x.shape[-1] * x.shape[-2]))(x)

    for i in range(rnn_layers):
        x = Bidirectional(
            LSTM(units, activation=act, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(drop_out)(x)

    bn_rnn = BatchNormalization()(x)

    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred, name="custom_model")

    return model

def build_model(output_dim, custom_model, preprocess_model, calc=None):

    input_audios = Input(name='the_input', shape=(None,))
    pre = preprocess_model(input_audios)
    pre = tf.squeeze(pre, [3])

    y_pred = custom_model(pre)
    model = Model(inputs=input_audios, outputs=y_pred, name="model_builder")
    model.output_length = calc

    return model

def predict(audio):
    pred_audios = tf.convert_to_tensor([audio])
    y_pred = model.predict(pred_audios)

    input_shape = tf.keras.backend.shape(y_pred)
    input_length = tf.ones(shape=input_shape[0]) * tf.keras.backend.cast(input_shape[1], 'float32')
    prediction = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=False)[0][0]

    pred = K.eval(prediction).flatten().tolist()
    pred = [i for i in pred if i != -1]

    hypothesis   = int_sequence_to_text(pred)
    return hypothesis

sample_rate = 22050
fft_size = 1024
frame_step = 512
n_mels = 128
output_dim = len(char_map) + 2

preprocess_model = preprocessin_model(sample_rate, fft_size, frame_step, n_mels)
speech_model = conv_rnn(n_mels, output_dim = output_dim)
model = build_model(output_dim, speech_model, preprocess_model)
model.load_weights('src/pages/cnn_rnn5.h5')


def record(duration=5, fs=22050):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording

def write():
    st.header("1. Record your own voice")
    if st.button(f"Click to Record"):
        record_state = st.text("Recording...")
        duration = 6  # seconds
        recorded_audio = record(duration)

        record_state.text(f"Audio recording done!")
        wavio.write('images/sample.wav', recorded_audio, 22050, sampwidth=2)
        st.audio('images/sample.wav', format='audio/wav')

        if st.button('Predict'):
            # st.write(wav_file)
            # audio, fs = librosa.load('images/sample.wav')
            pred = predict(recorded_audio)
            if pred:
                st.write(pred)
            else:
                st.write("Could not predict")


    st.header("2. Choose an audio record")
    wav_file = st.file_uploader("Upload wav file", type=['wav'])
    if (wav_file):
        st.audio(wav_file, format='audio/wav')

        if st.button('Predict'):
            # st.write(wav_file)
            audio, _ = librosa.load(wav_file)
            pred = predict(audio)
            if pred:
                st.write(pred)
            else:
                st.write("Could not predict")