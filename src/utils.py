import string
import librosa
import numpy as np

from typing import List

def stretch(x: np.ndarray, factor: float, nfft: int = 2048) -> np.ndarray:
    '''
    @author: Gagandeep Singh, 29 Oct, 2018
    https://github.com/gaganbahga/time_stretch
    
    stretch an audio sequence by a factor using FFT of size nfft converting to frequency domain
    :param x: np.ndarray, audio array in PCM float32 format
    :param factor: float, stretching or shrinking factor, depending on if its > or < 1 respectively
    :return: np.ndarray, time stretched audio
    '''
    stft = librosa.core.stft(x, n_fft=nfft).transpose()  # i prefer time-major fashion, so transpose
    stft_rows = stft.shape[0]
    stft_cols = stft.shape[1]

    times = np.arange(0, stft.shape[0], factor)  # times at which new FFT to be calculated
    hop = nfft/4                                 # frame shift
    stft_new = np.zeros((len(times), stft_cols), dtype=np.complex_)
    phase_adv = (2 * np.pi * hop * np.arange(0, stft_cols))/ nfft
    phase = np.angle(stft[0])

    stft = np.concatenate( (stft, np.zeros((1, stft_cols))), axis=0)

    for i, time in enumerate(times):
        left_frame = int(np.floor(time))
        local_frames = stft[[left_frame, left_frame + 1], :]
        right_wt = time - np.floor(time)                        # weight on right frame out of 2
        local_mag = (1 - right_wt) * np.absolute(local_frames[0, :]) + right_wt * np.absolute(local_frames[1, :])
        local_dphi = np.angle(local_frames[1, :]) - np.angle(local_frames[0, :]) - phase_adv
        local_dphi = local_dphi - 2 * np.pi * np.floor(local_dphi/(2 * np.pi))
        stft_new[i, :] =  local_mag * np.exp(phase*1j)
        phase += local_dphi + phase_adv

    return librosa.core.istft(stft_new.transpose())

def meow_stretch(
    x: np.ndarray, character_len: int, 
    init_factor: float = 0.3, add_factor: float = 0.2,
    power_factor: float = 0.8, nfft: int = 2048
) -> np.ndarray:
    '''
    Stretch the meows based on word length, with a reducing power to prevent incredibly long meows
    '''
    
    factor = init_factor + (add_factor * character_len) ** power_factor
    return stretch(x, 1/factor, nfft=nfft)

def get_word_lengths(text_input: str) -> List[int]:
    text_input = text_input.translate(str.maketrans('', '', string.punctuation))
    word_list = text_input.split()
    
    return [len(word) for word in word_list]
