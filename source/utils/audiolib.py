
import os
import numpy as np
import soundfile as sf
import glob
import librosa

from urllib.request import urlretrieve

EPS = np.finfo(float).eps

COEFS_SIG = np.array([9.651228012789436761e-01, 6.592637550310214145e-01,
                    7.572372955623894730e-02])
COEFS_BAK = np.array([-3.733460011101781717e+00,2.700114234092929166e+00,
                    -1.721332907340922813e-01])
COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01,
                    7.600269530243179694e-02])


def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def normalize_segmental_rms(audio, rms, target_level=-25):
    '''Normalize the signal to the target level
    based on segmental RMS'''
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def audioread(path, norm=False, start=0, stop=None, target_level=-25):
    '''Function to read audio'''

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(audio.shape) == 1:  # mono
        if norm:
            rms = (audio ** 2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms + EPS)
            audio = audio * scalar
    else:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0) / audio.shape[0]
        if norm:
            audio = normalize(audio, target_level)

    return audio, sample_rate


def audiowrite(destpath, audio, sample_rate=16000, norm=False, target_level=-25, \
               clipping_threshold=0.99, clip_test=False):
    '''Function to write audio'''

    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError("Clipping detected in audiowrite()! " + \
                             destpath + " file not written to disk.")

    if norm:
        audio = normalize(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio / max_amp * (clipping_threshold - EPS)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return


def add_clipping(audio, max_thresh_perc=0.8):
    '''Function to add clipping'''
    threshold = max(abs(audio)) * max_thresh_perc
    audioclipped = np.clip(audio, -threshold, threshold)
    return audioclipped


def snr_mixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various SNR levels'''
    cfg = params['cfg']
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))

    # Normalizing to -25 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean ** 2).mean() ** 0.5

    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise ** 2).mean() ** 0.5

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def segmental_snr_mixer(clean, noise, snr, target_level=-25, clipping_threshold=0.99, target_lower=-35, target_upper=-15):
    '''Function to mix clean speech and noise at various segmental SNR levels'''

    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))
    clean = clean / (max(abs(clean)) + EPS)
    noise = noise / (max(abs(noise)) + EPS)

    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    noisy_rms_level = np.random.randint(target_lower, target_upper)
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = 20 * np.log10((noise_win ** 2).mean() + EPS)
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs) != 0:
        noise_rms = (noise_active_segs ** 2).mean() ** 0.5
    else:
        noise_rms = EPS

    if len(clean_active_segs) != 0:
        clean_rms = (clean_active_segs ** 2).mean() ** 0.5
    else:
        clean_rms = EPS

    return clean_rms, noise_rms


def activitydetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    '''Return the percentage of the time the audio signal is above an energy threshold'''

    audio = normalize(audio, target_level)
    window_size = 50  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win ** 2) + EPS)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (1 - alpha_att)
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (1 - alpha_rel)

        if smoothed_energy_prob > energy_thresh:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def resampler(input_dir, target_sr=16000, ext='*.wav'):
    '''Resamples the audio files in input_dir to target_sr'''
    files = glob.glob(f"{input_dir}/" + ext)
    for pathname in files:
        print(pathname)
        try:
            audio, fs = audioread(pathname)
            audio_resampled = librosa.core.resample(audio, fs, target_sr)
            audiowrite(pathname, audio_resampled, target_sr)
        except:
            continue


def audio_segmenter(input_dir, dest_dir, segment_len=10, ext='*.wav'):
    '''Segments the audio clips in dir to segment_len in secs'''
    files = glob.glob(f"{input_dir}/" + ext)
    for i in range(len(files)):
        audio, fs = audioread(files[i])

        if len(audio) > (segment_len * fs) and len(audio) % (segment_len * fs) != 0:
            audio = np.append(audio, audio[0: segment_len * fs - (len(audio) % (segment_len * fs))])
        if len(audio) < (segment_len * fs):
            while len(audio) < (segment_len * fs):
                audio = np.append(audio, audio)
            audio = audio[:segment_len * fs]

        num_segments = int(len(audio) / (segment_len * fs))
        audio_segments = np.split(audio, num_segments)

        basefilename = os.path.basename(files[i])
        basename, ext = os.path.splitext(basefilename)

        for j in range(len(audio_segments)):
            newname = basename + '_' + str(j) + ext
            destpath = os.path.join(dest_dir, newname)
            audiowrite(destpath, audio_segments[j], fs)


def standardize_audio_size(audio, fs, input_len):
    """
    Adjust audio size to be of size fs * self.input_len_second
    If len(audio) > fs * self.input_len_second, sample a sub audio clip of size fs * self.input_len_second
    If len(audio) < fs * self.input_len_second, pad the audio clip with itself
    :param audio: np.array
    :param fs: int, sampling rate
    :param input_len: input len in second
    :return:
    """
    audio = np.tile(audio, np.ceil(fs * input_len / audio.shape[0]).astype('int32'))

    start_idx = np.random.randint(0, len(audio) - input_len * fs + 1)
    end_idx = start_idx + int(input_len * fs)

    return audio[start_idx:end_idx]

def audio_logpowspec(audio, nfft=320, hop_length=160, sr=16000):
    """
    Log-power specturm for each time window
    :param audio: audio numpy array
    :param nfft:
    :param hop_length: int, window hop
    :param sr: int, sample rate
    :return: (time, freq) spectogram of size (nframes , 1 + nfft/2)
    """
    powspec = (np.abs(librosa.core.stft(audio, n_fft=nfft, hop_length=hop_length)))**2
    logpowspec = np.log10(np.maximum(powspec, 10**(-12)))
    return logpowspec.T


def load_audio_file(clip_url, temp_folder='./temp', input_length=9.0, remove=True, standardize=True):
    """

    :param clip_url: path to audio clip
    :return: np.array, int sample rate
    """
    os.makedirs(temp_folder, exist_ok=True)
    local = True

    if clip_url.startswith('https:'):
        local = False
        local_url = os.path.basename(clip_url)
        local_url = os.path.join(temp_folder, local_url)

        try:
            local_name, _ = urlretrieve(clip_url, local_url)
            print(f'Loading file {clip_url}')

        except:
            print(f'Error when reading file {clip_url}')
            return None, None
    else:
        local_name = clip_url

    audio, fs = sf.read(local_name)

    if standardize:
        audio = standardize_audio_size(audio, fs, input_length)

    if remove and local == False:
        os.remove(local_name)

    return audio, fs


def audio_melspec(audio, n_mels=64, window_len=400, hop_length=160, sr=16000, center=True, window='hann'):
    """
    MelLog-power specturm for each time window
    :param audio: audio numpy array
    :param window_len:
    :param hop_length: int, window hop
    :param sr: int, sample rate
    :return: (time, freq) spectogram of size (nframes , 1 + nfft/2)
    """
    n_fft = 2 ** int(np.ceil(np.log(window_len) / np.log(2.0)))

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
                                                  hop_length=hop_length,
                                                  n_mels=n_mels,
                                                  center=center,
                                                  win_length=window_len,
                                                  window=window
                                                  )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=1, amin=1e-10)
    log_mel_spec = log_mel_spec.astype(np.float32)

    return log_mel_spec.T


def get_one_zero_label(label_codes, tag_mapping, num_labels=521):
    """
    Convert labels code into one-hot vector
    :param label_codes:
    :param tag_mapping: dictionary with key equal to label codes
    :param num_class:
    :return:
    """
    label_codes = label_codes.split(',')
    labels_numerical = [int(tag_mapping[lab]['tag_numerical']) for lab in label_codes if lab in tag_mapping]
    label_one_hot = np.zeros(num_labels)

    for lab in labels_numerical:
        label_one_hot[lab] = 1

    return label_one_hot


def infer_mos(audio, fs, input_length, session_sig, session_bak_ovr):
    """
     Compute mos_sig, mos_bak, mos_ovr predicted by models in session_sig, session_bak_ovr
    :param audio:
    :param fs:
    :param input_length:
    :param session_sig:
    :param session_bak_ovr:
    :return:
    """

    input_length = input_length

    num_hops = int(np.floor(len(audio) / fs) - input_length) + 1
    hop_len_samples = fs
    predicted_mos_sig_seg = []
    predicted_mos_bak_seg = []
    predicted_mos_ovr_seg = []

    for idx in range(num_hops):
        audio_seg = audio[int(idx * hop_len_samples): int((idx + input_length) * hop_len_samples)]
        input_features = np.array(audio_logpowspec(audio=audio_seg, sr=fs)).astype('float32')[np.newaxis, :, :]

        # sig predictions
        onnx_inputs_sig = {inp.name: input_features for inp in session_sig.get_inputs()}
        mos_sig = np.polynomial.polynomial.polyval(session_sig.run(None, onnx_inputs_sig), COEFS_SIG)

        # bak_mos predicitions
        onnx_inputs_bak_ovr = {inp.name: input_features[:, :-1, :] for inp in session_bak_ovr.get_inputs()}
        mos_bak_ovr = session_bak_ovr.run(None, onnx_inputs_bak_ovr)
        mos_bak = np.polynomial.polynomial.polyval(mos_bak_ovr[0][0][1], COEFS_BAK)
        mos_ovr = np.polynomial.polynomial.polyval(mos_bak_ovr[0][0][2], COEFS_OVR)

        predicted_mos_sig_seg.append(mos_sig)
        predicted_mos_bak_seg.append(mos_bak)
        predicted_mos_ovr_seg.append(mos_ovr)

    return np.mean(predicted_mos_sig_seg), np.mean(predicted_mos_bak_seg), np.mean(predicted_mos_ovr_seg)