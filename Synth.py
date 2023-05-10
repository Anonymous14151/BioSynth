from abc import ABC, abstractmethod
import itertools

import neurokit2 as nk

import numpy as np
import math

from scipy.io import wavfile
from scipy.signal import resample, butter, filtfilt

import librosa

import matplotlib.pyplot as plt

from pygame import mixer

"""
This is a code totally based on the very good tutorials from Alan. You can find them here:
https://python.plainenglish.io/making-a-synth-with-python-oscillators-2cb8e68e9c3b
"""


class SynthWave:
    @staticmethod
    def wave_to_file(wav, wav2=None, fname="temp.wav", amp=0.1, sample_rate=44100):
        wav = np.array(wav)
        wav = np.int16(wav * amp * (2 ** 15 - 1))

        if wav2 is not None:
            wav2 = np.array(wav2)
            wav2 = np.int16(wav2 * amp * (2 ** 15 - 1))
            wav = np.stack([wav, wav2]).T

        wavfile.write(fname, sample_rate, wav)

    @staticmethod
    def get_seq(osc, notes=["C4", "E4", "G4"], note_lens=[0.5, 0.5, 0.5], SR=44100):
        samples = []
        if(len(osc)>1):
            #iter over list of oscillators
            [iter(osc_i) for osc_i in osc]
            for note, note_len in zip(notes, note_lens):
                #change freq individually
                for osc_i in osc:
                    osc_i.freq = librosa.note_to_hz(note)
                for _ in range(int(SR * note_len)):
                    #add the waves
                    samples.append(sum(next(osc_i) for osc_i in osc) / len(osc))
        else:
            osc = iter(osc[0])
            for note, note_len in zip(notes, note_lens):
                osc.freq = librosa.note_to_hz(note)
                for _ in range(int(SR * note_len)):
                    samples.append(next(osc))
        return samples

    @staticmethod
    def get_modulated_seq(mod_osc, notes=["C4", "E4", "G4"], note_lens=[0.5, 0.5, 0.5], SR=44100):
        """
        A way to modulate the sound of sequences of notes with their duration
        :param mod_osc:
        :param notes:
        :param note_lens:
        :param SR:
        :return:
        """
        samples = []
        mod_osc = iter(mod_osc)
        for note, note_len in zip(notes, note_lens):
            mod_osc.oscillator.freq = librosa.note_to_hz(note)
            for _ in range(int(SR * note_len)):
                samples.append(next(mod_osc))

        return samples


class Oscillator(ABC):
    def __init__(self, freq=10, phase=0, amp=1, sample_rate=44100, wave_range=(-1, 1)):
        self._freq = freq
        self._phase = phase
        self._amp = amp
        self._sample_rate = sample_rate
        self._wave_range = wave_range

        # properties that can be changed
        self._f = freq
        self._p = phase
        self._a = amp

    """
    Create properties of frequency, phase and amplitude. This enables that we create setters and getters to update
    such values without loosing the original value defined at the beggining. We keep the fundamental values, and can 
    adapt these values over time.
    """

    @property
    def init_freq(self):
        return self._freq

    @property
    def init_amp(self):
        return self._amp

    @property
    def init_phase(self):
        return self._phase

    """
    We can create the setters of the properties.
    These setters change the value of a parameter, but we can keep the fundamental value.
    The idea is that when a key is pressed, it starts with the fundamental frequency ._freq, but if the key is kept it can 
    alter the frequency ._f and modulate it. The same happens with the other parameters.
    We will define a __init__ and __next__ functions. The __init__ method will be used to instantiate the key press 
    and fundamental properties, while the __next__ method will call the setter and the property ._f and kept being called
    as long as the key is pressed.
    """

    @property
    def freq(self):
        return self._f

    @freq.setter
    def freq(self, value):
        self._f = value
        self._post_freq_set()

    @property
    def amp(self):
        return self._a

    @amp.setter
    def amp(self, value):
        self._a = value
        self._post_amp_set()

    @property
    def phase(self):
        return self._p

    @phase.setter
    def phase(self, value):
        self._p = value
        self._post_phase_set()

    def _post_freq_set(self):
        pass

    def _post_amp_set(self):
        pass

    def _post_phase_set(self):
        pass

    @abstractmethod
    def _initialize_osc(self):
        pass

    @staticmethod
    def squish_val(val, min_val=0, max_val=1):
        """
        static method squish_val to normalize values of the oscillator into a specific range
        """
        return (((val + 1) / 2) * (max_val - min_val)) + min_val

    @abstractmethod
    def __next__(self):
        return None

    def __iter__(self):
        self.freq = self._freq
        self.phase = self._phase
        self.amp = self._amp
        self._initialize_osc()
        return self

class SineOscillator(Oscillator):
    """
    Sine Wave generator
    """
    # inheritance of Oscillator class
    def _post_freq_set(self):
        self._step = (2 * math.pi * self._f) / self._sample_rate

    def _post_phase_set(self):
        self._p = (self._p / 360) * 2 * math.pi

    def _initialize_osc(self):
        self._i = 0

    def __next__(self):
        val = math.sin(self._i + self._p)
        self._i = self._i + self._step
        if self._wave_range is not (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

class SquareOscillator(SineOscillator):
    def __init__(self, freq=440, phase=0, amp=1, \
                 sample_rate=44_100, wave_range=(-1, 1), threshold=0):
        super().__init__(freq, phase, amp, sample_rate, wave_range)
        self.threshold = threshold

    def __next__(self):
        val = math.sin(self._i + self._p)
        self._i = self._i + self._step
        if val < self.threshold:
            val = self._wave_range[0]
        else:
            val = self._wave_range[1]
        return val * self._a

class SawtoothOscillator(Oscillator):
    def _post_freq_set(self):
        self._period = self._sample_rate / self._f
        self._post_phase_set

    def _post_phase_set(self):
        self._p = ((self._p + 90) / 360) * self._period

    def _initialize_osc(self):
        self._i = 0

    def __next__(self):
        div = (self._i + self._p) / self._period
        val = 2 * (div - math.floor(0.5 + div))
        self._i = self._i + 1
        if self._wave_range is not (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

class TriangleOscillator(SawtoothOscillator):
    def __next__(self):
        div = (self._i + self._p) / self._period
        val = 2 * (div - math.floor(0.5 + div))
        val = (abs(val) - 0.5) * 2
        self._i = self._i + 1
        if self._wave_range is not (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

class WaveAdder:
    """
    This is a class for Additive Synthesis, which enables the addition of multiple sine waves together
    """

    def __init__(self, *oscillators):
        self.oscillators = oscillators
        self.n = len(oscillators)

    def __iter__(self):
        [iter(osc) for osc in self.oscillators]
        return self

    def __next__(self):
        return sum(next(osc) for osc in self.oscillators) / self.n


class ModulatedOscillator:
    def __init__(self, oscillator, *modulators, amp_mod=None, freq_mod=None, phase_mod=None):
        self.oscillator = oscillator
        self.modulators = modulators #list
        self.amp_mod = amp_mod
        self.freq_mod = freq_mod
        self.phase_mod = phase_mod
        self._modulators_count = len(modulators)

    def _modulate(self, mod_vals):
        if(self.amp_mod is not None):
            new_amp = self.amp_mod(self.oscillator.init_amp, mod_vals[0])
            self.oscillator.amp = new_amp

        if(self.freq_mod is not None):
            if self._modulators_count == 2:
                mod_val = mod_vals[1]
            else:
                mod_val = mod_vals[0]
            new_freq = self.freq_mod(self.oscillator.init_freq, mod_val)
            self.oscillator.freq = new_freq

        if (self.phase_mod is not None):
            if self._modulators_count == 3:
                mod_val = mod_vals[2]
            else:
                mod_val = mod_vals[-1]
            new_phase = self.phase_mod(self.oscillator.init_phase, mod_val)
            self.oscillator.phase = new_phase

    def trigger_release(self):
        """
        this function helps to control the keys pressed and how the sound plays
        :return:
        """
        tr = "trigger_release"
        for modulator in self.modulators:
            if(hasattr(modulator, tr)):
                modulator.trigger_release()
        if(hasattr(self.oscillator, tr)):
            self.oscillator.trigger_release()

    @property
    def ended(self):
        e = "ended"
        ended = []
        for modulator in self.modulators:
            if(hasattr(modulator, e)):
                ended.append(modulator.ended)
        if hasattr(self.oscillator, e):
            ended.append(self.oscillator.ended)
        return all(ended)


    def __iter__(self):
        iter(self.oscillator)
        self.modulators = [iter(modulator) for modulator in self.modulators]
        return self

    def __next__(self):
        mod_vals = [next(modulator) for modulator in self.modulators]
        self._modulate(mod_vals)
        return next(self.oscillator)

class ADSREnvelope:
    def __init__(self, attack_duration=0.05, decay_duration=0.2, sustain_level=0.7, release_duration=0.3, sr = 44100):
        self.attack_duration = attack_duration
        self.decay_duration = decay_duration
        self.sustain_level = sustain_level
        self.release_duration = release_duration
        self.sample_rate = sr

    def get_ads_stepper(self):
        steppers = []
        if self.attack_duration > 0:
            steppers.append(itertools.count(start=0, step= 1/(self.attack_duration*self.sample_rate)))
        if self.decay_duration > 0:
            steppers.append(itertools.count(start=0, step=-(1-self.sustain_level)/(self.decay_duration*self.sample_rate)))

        while True:
            l = len(steppers)
            if l > 0:
                val = next(steppers[0])
                if l == 2 and val > 1:
                    steppers.pop(0)
                    val = next(steppers[0])
                elif l == 1 and val < self.sustain_level:
                    steppers.pop(0)
                    val = self.sustain_level
            else:
                val = self.sustain_level

            yield val

    def get_r_stepper(self):
        val = 1
        if(self.release_duration > 0):
            release_step = -self.val /(self.release_duration*self.sample_rate)
            stepper = itertools.count(start=1, step=release_step)
        else:
            val = -1

        while True:
            if val <= 0:
                self.ended = True
                val = 0
            else:
                val = next(stepper)

            yield val

    def __iter__(self):
        self.val = 0
        self.ended = False
        self.stepper = self.get_ads_stepper()
        return self

    def __next__(self):
        self.val = next(self.stepper)
        return self.val

    def trigger_release(self):
        self.stepper = self.get_r_stepper()


class Biosig():
    def __init__(self, sig, type="emg", sr=1000):
        self.sig = sig
        self.sig_type = type
        self.sr = sr
        self.music_sr = 44100

        if(self.sig_type == "ecg"):
            #normalize
            self.s_norm = (self.sig - np.min(self.sig)) / (np.max(self.sig) - np.min(self.sig))
            #resample to sound wave sampling rate
            self.sig_proc = self.resample(self.s_norm)
            #detect peaks
            _, info = nk.ecg.ecg_peaks(self.sig_proc, sampling_rate=self.music_sr)
            self.peaks_pos = [0] + list(info["ECG_R_Peaks"]) + [2 * info["ECG_R_Peaks"][-1] - info["ECG_R_Peaks"][-2]]
            #get rr intervals
            self.rr_ = np.diff(self.peaks_pos) / self.music_sr
            self.rr_avg = [np.mean(self.rr_[i:i + 5]) for i in range(0, len(self.rr_ - 4))]
            self.rr_avg_bpm = [60 / i for i in self.rr_avg]
            self.rr_avg_res = self.resample(self.rr_avg_bpm)

        elif(self.sig_type == "emg"):
            self.env = self.resample(self._normalize(self.low_pass(np.abs(self.sig - np.mean(self.sig)), 2)))

    def process(self):
        self.normalize()
        self.sig_proc = self.resample(self.s_norm)

    def resample(self, sig_i):
        return resample(sig_i, len(self.sig) * 44100 // self.sr)  # resample to audio SR

    def low_pass(self, s, fc, order=2):
        # butterworth filter
        b, a = butter(order, fc, btype="low", fs=self.sr)
        return filtfilt(b, a, s)

    def normalize(self):
        self.s_norm = (self.sig-np.min(self.sig))/(np.max(self.sig)-np.min(self.sig))

    @staticmethod
    def _normalize(s):
        return (s-np.min(s))/(np.max(s)-np.min(s))

    @staticmethod
    def amp_mod(init_amp, env):
        return env * init_amp

    @staticmethod
    def freq_mod_amp(init_freq, env):
        return 0.5*init_freq + 2*init_freq*env

    def ecg_peaks(self):
        if(self.sig_type=="ecg"):
            self.process()
            _, info = nk.ecg.ecg_peaks(self.sig_proc, sampling_rate=self.music_sr)
            self.peaks_pos = [0] + list(info["ECG_R_Peaks"]) + [2*info["ECG_R_Peaks"][-1] - info["ECG_R_Peaks"][-2]]
        else:
            self.peaks_pos = [] #empty peaks list for other signal types

def map_value(y, min_val, max_val, min_res, max_res):
    return min_res + ((y-min_val)/(max_val-min_val))*(max_res-min_res)

def amp2keys(y_data, max_y, min_y):
    note_names = ['C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3',
                  'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
                  'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5'
                  ]

    n_notes = len(note_names)
    notes_seq = []
    for i in range(len(y_data)):
        note_index = round(map_value(y_data[i], min_y, max_y, 0, n_notes-1))
        notes_seq.append(note_names[note_index])

    return notes_seq

def createMelody(list_keys, list_dur, dur_sec):
    s = list_keys
    l = [dur_sec*(dur_i/np.sum(list_dur)) for dur_i in list_dur]

    return s, l

# def createMelody(dur):
#     s1 = ["C4", "E4", "G4", "B4"] * dur
#     l1 = [0.25, 0.25, 0.25, 0.25] * dur
#
#     s2 = ["C3", "E3", "G3"] * dur
#     l2 = [0.333334, 0.333334, 0.333334] * dur
#
#     s3 = ["C2", "G2"] * dur
#     l3 = [0.5, 0.5] * dur
#
#     return s1, l1


if __name__ == "__main__":
    #try with the ecg signal
    ecg = nk.ecg_simulate(duration=60, sampling_rate=1000, heart_rate=60, heart_rate_std=5)
    signals, info = nk.ecg_process(ecg, sampling_rate=1000)
    s = np.array(signals["ECG_Clean"])
    peaks_ = np.array(signals["ECG_R_Peaks"])

    #try amplitude modulation with emg envelope
    # s = np.loadtxt("data/SampleEMG.txt")[:, 2]
    synth_obj = SynthWave()
    # biosig_emg = Biosig(s)
    # biosig_emg.envelope() #compute envelope
    biosig_ecg = Biosig(s, type="ecg")
    biosig_ecg.process()
    biosig_ecg.ecg_rr() #calculate the heart rate (with 5 beats of average)


    #modulate with EMG amplitude
    gen = ModulatedOscillator(
        TriangleOscillator(freq=librosa.note_to_hz("G3")),
        biosig_ecg.sig_proc,
        amp_mod=biosig_ecg.amp_mod)

    # gen = ModulatedOscillator(
    #     SineOscillator(freq=librosa.note_to_hz("G4")),
    #     ADSREnvelope(attack_duration=1, decay_duration=2, sustain_level=0.1, release_duration=1),
    #     amp_mod=biosig_ecg.amp_mod
    # )

    gen2 = SineOscillator(freq=librosa.note_to_hz("C4"))

    iter(gen)
    wav = [next(gen) for _ in range(len(biosig_ecg.sig_proc))]
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(biosig_ecg.sig_proc)
    ax2.plot(biosig_ecg.rr_avg_bpm)
    plt.show()

    # iter(gen2)
    # wav2 = [next(gen2) for _ in range(len(wav))]
    notes_seq = amp2keys(biosig_ecg.rr_avg_bpm, 90, 40)
    dur_sec = len(wav)/44100
    s1, l1 = createMelody(notes_seq, biosig_ecg.rr_, dur_sec)

    wav2 = synth_obj.get_seq([gen2], notes=s1, note_lens=l1)

    print(len(wav))
    print(len(wav2))
    # wav = synth_obj.get_modulated_seq(gen, notes=s1, note_lens=l1)
    synth_obj.wave_to_file(wav[:len(wav2)], wav2, fname="ecg_sine.wav")
    # synth_obj.wave_to_file(wav2, fname="ecg_sine.wav")

    mixer.init()
    mixer.music.load("sound_output/ecg_sine.wav")
    mixer.music.play()

    while True:
        userInput = input(" ")

        if(userInput == "e"):
            mixer.music.stop()
            break

    # # wav = [next(gen) for _ in range(44100 * 4)]
    #
    # synth_obj = SynthWave()
    #
    # wav = synth_obj.get_seq([SineOscillator(), TriangleOscillator()], notes=s1, note_lens=l1) +\
    #     synth_obj.get_seq([SineOscillator(), SawtoothOscillator()], notes=s2, note_lens=l2)
    #
    # synth_obj.wave_to_file(wav, fname="prelude_one.wav")