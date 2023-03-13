from pedalboard import Pedalboard, Chorus, Reverb, Distortion, Phaser, Delay, load_plugin
from pedalboard.io import AudioFile
import os
import numpy as np
import librosa
import pyloudnorm as pyln

# STEPS
# 1. start with only 3 effects: overdrive, delay, reverb
# 2. adding 2 more effects: chorus, tremolo

UNPROCESSED_PATH_4_GUITARS = r'G:\tesi_4maggio_22\dataset_michele\multilabel\unprocessed_4guit'
PROCESSED_PATH_4_GUITARS = r'G:\tesi_4maggio_22\dataset_michele\multilabel'
LOUDNESS_LEVEL = -26

# plugin_paths
OVERDRIVE_PATH = r"G:\tesi_4maggio_22\a__plugins\The Klone.vst3"
FLANGER_PATH = r"G:\tesi_4maggio_22\a__plugins\modulation\MFlanger.vst3"
VIBRATO_PATH = r"G:\tesi_4maggio_22\a__plugins\modulation\MVibrato.vst3"
TREMOLO_PATH = r"G:\tesi_4maggio_22\a__plugins\modulation\MTremolo.vst3"

'''
def export_reverb_sample(file, wet_value, unprocessed_audio, processed_path, sr):

    reverb_path = os.path.join(processed_path, "reverb")
    if not os.path.exists(reverb_path):
        os.mkdir(reverb_path)
        print("New folder created: ", reverb_path)

    processed_audio = add_reverb_normalized(unprocessed_audio, sr, wet_level=wet_value)
    sample_path = os.path.join(reverb_path, f"{file[:-4]}_reverb_{wet_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def add_reverb_normalized(unprocessed_audio, sr, wet_level):
    board = Pedalboard([Reverb(wet_level=wet_level)])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio
'''


def test_numpy_reshape_function():
    pedalboard = np.array([[1, 2, 3, 4, 5, 6, 7]])

    reshaped = np.reshape(pedalboard, np.shape(pedalboard)[1])
    reshaped = reshaped + 1

    reshaped_2 = np.reshape(reshaped, (1, np.shape(pedalboard)[1]))

    print(f'- pedalboard shape:{np.shape(pedalboard)}  print: {pedalboard}')
    print(f'- reshaped shape:  {np.shape(reshaped)}    print: {reshaped}')
    print(f'- reshaped2 shape: {np.shape(reshaped_2)}  print: {reshaped_2}')


def normalize_loudness(audio_file, sr, loudness_level, verbose=False):
    # reshape audio file (1, 507150) -> (507150,) that is how Pedalboard loads vs how librosa loads
    audio_file_reshape = np.reshape(audio_file, np.shape(audio_file)[1])

    # create BS.1770 meter and get audio loudness
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_file_reshape)

    # loudness normalize to -26 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(audio_file_reshape, loudness, loudness_level)

    # get back to the shape pedalboard works with: (1, 507150)
    audio_file_normalized = np.reshape(loudness_normalized_audio, (1, np.shape(audio_file)[1]))

    if verbose:
        print(f"\nDETAILS OF normalized_audio FUNCTION\n"
              f"- shape of input audio sample: {np.shape(audio_file)}\n"
              f"- shape after reshape function: {np.shape(audio_file_reshape)}\n"
              f"- shape after normalization: {np.shape(loudness_normalized_audio)}\n"
              f"- output shape after reshape: {np.shape(audio_file_normalized)}")

    return audio_file_normalized


def get_file_name_with_plugin_list(audio_file_name, effects_list):
    # get a file name like les_bridge_fing01__101.wav
    effect_list_string = "__"
    for element in effects_list:
        effect_list_string = effect_list_string + f"{element}"

    new_file_name = audio_file_name[:-4] + effect_list_string + ".wav"

    return new_file_name


def export_processed_audio(normalized_processed_audio, sr, audio_file_name, processed_files_path, effects_list,
                           verbose=False):
    if verbose:
        print(f"\nFUNCTION: export_processed_audio\n"
              f"- exported file path: {processed_files_path}\n"
              f"- audio file name: {audio_file_name}")

    # get the path including the file name and plugin list(ex. 'G:...\les_bridge_fing01__011.wav')
    audio_file_name_with_plugin_list = get_file_name_with_plugin_list(audio_file_name, effects_list)
    complete_audio_path = os.path.join(processed_files_path, audio_file_name_with_plugin_list)

    with AudioFile(complete_audio_path, 'w', sr, normalized_processed_audio.shape[0]) as f:
        f.write(normalized_processed_audio)


def process_file(unprocessed_audio, sr, effects_list, overdrive_effect, tremolo_effect):
    # normalize audio file (from -23 to -26)
    normalized_unprocessed_audio = normalize_loudness(unprocessed_audio, sr, LOUDNESS_LEVEL, verbose=False)

    # create the plugin list (ex: [overdrive_effect, Reverb] )

    # effect order: overdrive, chorus, tremolo, delay, reverb
    plugin_list = []
    if effects_list[0] == 1:
        plugin_list.append(overdrive_effect)
    if effects_list[1] == 1:
        plugin_list.append(Chorus())
    if effects_list[2] == 1:
        plugin_list.append(tremolo_effect)
    if effects_list[3] == 1:
        plugin_list.append(Delay())
    if effects_list[4] == 1:
        plugin_list.append(Reverb())

    board = Pedalboard(plugin_list)
    effected_audio = board(normalized_unprocessed_audio, sr)
    normalized_processed_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)

    return normalized_processed_audio


def print_audio_information(unprocessed_audio, sr, loaded_with):
    print(f'\nAUDIO FILE LOADED WITH: {loaded_with}\n'
          f'- type: {type(unprocessed_audio)}\n'
          f'- shape: {np.shape(unprocessed_audio)}\n'
          f'- sample rate: {sr}')


def load_audio_file(file_path):
    with AudioFile(file_path, 'r') as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate
    return audio, samplerate


def create_directory_for_processed_files(processed_path):
    # get the full path and create a new folder
    new_path = os.path.join(processed_path, "processed_5effects_vary_params")
    os.mkdir(new_path)
    print("Processed guitar folder created: ", new_path)

    return new_path


def get_plugin_list_combinations():
    combinations = []
    for index_1 in range(2):
        for index_2 in range(2):
            for index_3 in range(2):
                for index_4 in range(2):
                    for index_5 in range(2):
                        combinations.append([index_1, index_2, index_3, index_4, index_5])

    return combinations


def process_files(path, files, overdrive_effect, tremolo_effect):
    print_info = True
    process_only_one_file = 0

    # create a folder for processed files
    processed_files_path = create_directory_for_processed_files(PROCESSED_PATH_4_GUITARS)

    for audio_file_name in files:

        # get full path (ex: G:\tesi_4maggio_22\dataset_michele\...\tele_neck_pick14.wav)
        file_path = os.path.join(path, audio_file_name)

        # load audio files and verify information (info for only one sample)
        unprocessed_audio, sr = load_audio_file(file_path)
        if print_info:
            print_audio_information(unprocessed_audio, sr, 'Pedalboard')
            print_info = False

        # process a single file

        # get possible combinations of plugins (es [ [0,0,1,0,0], [1,0,1,1,1], ... ])
        plugin_list_combinations = get_plugin_list_combinations()

        # process one file for each combination of plugins
        for list_combination in plugin_list_combinations:
            normalized_processed_audio = process_file(unprocessed_audio, sr, list_combination, overdrive_effect, tremolo_effect)

            # export audio file
            export_processed_audio(normalized_processed_audio, sr, audio_file_name,
                                   processed_files_path, list_combination, verbose=False)

        # if process_only_one_file < 2:
        #   process_only_one_file += 1


def create_dataset(unprocessed_path):
    # load external audio plugins
    overdrive_effect = load_plugin(OVERDRIVE_PATH)
    tremolo_effect = load_plugin(TREMOLO_PATH)

    for path, folders, files in os.walk(unprocessed_path):
        # verify correct folder
        print(f'\nSTART MULTILABEL DATASET CREATION\n'
              f'- unprocessed samples path: {path}\n'
              f'- folders in this path: {folders}\n'
              f'- audio files in this path (few examples): {files[:4]}')

        # how many audio files we have
        print('- how many audio files:', len(files))

        # process each song separately
        process_files(path, files, overdrive_effect, tremolo_effect)


if __name__ == '__main__':

    create_dataset(UNPROCESSED_PATH_4_GUITARS)

    # next step
    # - create a folder to save the new files
    # - prosess one file with the 3 effects
