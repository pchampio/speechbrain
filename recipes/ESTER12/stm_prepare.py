"""
Data preparation.
Assume that transcription (.stm) is generated by Transcriber export.

Download (paid) ESTER1: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0241/
Download (paid) ESTER2: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0338/

Author
------
Pierre Champion
"""

import logging
import subprocess
import re
import glob
import os
import csv
import sys

from num2words import num2words
from tqdm import tqdm

from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

import soundfile


broken_encoding_replacements = {
        u'Ã ': u'à',
        u'Ã¢': u'â',
        u'Ã€': u'À',
        u'Ã§': u'ç',
        u'Ã‡': u'Ç',
        u'Ã©': u'é',
        u'Ã¨': u'è',
        u'Ãª': u'ê',
        u'Ã‰': u'É',
        u'Ãˆ': u'È',
        u'Ã®': u'î',
        u'Ã¯': u'ï',
        u'Ã±': u'ñ',
        u'Ã´': u'ô',
        u'Ã¹': u'ù',
        u'Ã»': u'û',
        }

import string
delset = string.punctuation
delset = delset.replace("'", "")
delset = delset.replace("%", "")
delset = delset.replace(",", "")

logger = logging.getLogger(__name__)
OPT_FILE = "opt_stm_prepare.pkl"
SAMPLERATE = 16000


logging.basicConfig(level=logging.INFO)


def prepare_stm(
        stm_directory,
        wav_directory,
        tr_splits,
        dev_splits,
        te_splits,
        save_folder,
        skip_prep=False,
        ignore_wav=False,
        ):
    """
    This class prepares the csv files for STM like dataset.

    Arguments
    ---------
    stm_directory : str
        Path to the folder where the original .stm files are stored (glob compatible)
    wav_directory : str
        Path to the folder where the original .wav files are stored (glob compatible)
    tr_splits : list
        List of train splits (regex from path) to prepare from ["/train*",r"other_data"]
    dev_splits : list
        List of dev splits (regex from path) to prepare from ["*/dev/*"].
    te_splits : list
        List of test splits (regex from path) to prepare from ["*/test/*"] -> create one test merged test dataset.
        Dict of List of test splits (regex from path) to prepare from {"test_ETAPE":["/ETAPE/test/*"], "test_ESTER2":["/ESTER2/test/*"], "test_ESTER1":["/ESTER1/test/*"]},
    save_folder : str
        The directory where to store the csv files.
    make_lm: bool
        If True, create arpa {3-4}grams LMs
    lm_gram_orders: list
        List of N grams order, defualt=[3,4]
    skip_prep: bool
        If True, data preparation is skipped.

    Example
    -------
    >>> prepare_stm(
    ...     "/corpus/**/[^\.ne_e2\.|\.ne\.|\.spk\.|part\.]*.stm",
    ...     "/corpus/**/*.wav",
    ...     [r"/train/", r"/train_trans_rapide/"],
    ...     [r"/dev/"],
    ...     [r"/test/"],
    ...     "./data_prep_out",
    ...     make_lm=True,
    ...)

    >>> prepare_stm(
    ...     "/corpus/ESTER[1-2]/**/[^\.ne_e2\.|\.ne\.|\.spk\.]*.stm",
    ...     "/corpns/ESTER[1-2]/**/*.wav",
    ...     [r"/train/", r"/train_trans_rapide/"],
    ...     [r"/dev/"],
    ...     [r"/test/"],
    ...     "./data_prep_out",
    ...)

    >>> prepare_stm(
    ...     "/corpns/ESTER2/**/[^\.ne_e2\.|\.ne\.|\.spk\.]*.stm",
    ...     "/corpus/ESTER2/**/*.wav",
    ...     [r"/train/", r"/train_trans_rapide/"],
    ...     [r"/dev/"],
    ...     [r"/test/"],
    ...     "./data_prep_out",
    ...)

    """

    if skip_prep:
        return

    os.makedirs(save_folder, exist_ok=True)

    conf = locals().copy()
    save_opt = os.path.join(save_folder, OPT_FILE)
    # Check if this phase is already done (if so, skip it)
    if skip(save_folder, conf, save_opt):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")


    stm_paths,  stm_exclude_match = custom_glob_filter(stm_directory)
    pbar = tqdm(stm_paths, bar_format='{desc} {percentage:3.0f}%')
    i = 0

    if not ignore_wav:
        wav_paths,  wav_exclude_match = custom_glob_filter(wav_directory)
        wav_paths_map = {normalize_wav_key(os.path.basename(wav_p)):wav_p  for wav_p in wav_paths}

    split_info = [(tr_splits, "train", []), (dev_splits, "dev", [])]
    if isinstance(te_splits, list):
        split_info.append((te_splits, "test", []))
    if isinstance(te_splits, dict):
        for te_splits_name, te_splits_paths in te_splits.items():
            split_info.append((te_splits_paths, te_splits_name, []))


    train_vocab = set()
    train_transcript_words = []

    for filename in pbar:
        if stm_exclude_match is not None:  # Exclude all paths with specified string
            if re.search(stm_exclude_match, filename):
                logger.debug(f"Skipping {filename}, as it is in the exclude match")
                continue

        split = "n"
        info = None
        for sp, id, _info in split_info:
            for tr in sp:
                if re.search(tr, filename):
                    split = id
                    info = _info
                    break

        if split == "n":
            logger.debug(f"Skipping {filename}, not associated to any split")
            continue

        d = (filename).ljust(80, ' ')
        i += 1
        pbar.set_description(f"Len: {str(i).rjust(4)} : Processing '{split}' : {d}")
        with open(filename, 'r') as file:
            data = file.readlines()
        for line in transform_lines(filterout_lines(data)):
            parts = line.split()
            wav_key = normalize_wav_key(parts[0]) 
            if not ignore_wav and wav_key not in wav_paths_map:
                logger.critical(f"Did not found wav '{wav_key}' for stm: '{filename}'")
                break
            else:
                text = text_transform(f"{' '.join(parts[6:])}")

                # No transcription, might be only rire/jingle anotation
                if text == "":
                    continue

                if not ignore_wav:
                    # wav file is not complete
                    audio_info = soundfile.info(wav_paths_map[wav_key])
                    startTime = float(parts[3])
                    endTime = float(parts[4])
                    wav_path = wav_paths_map[wav_key]
                    if startTime > audio_info.duration or int(endTime) > audio_info.duration:
                        logger.critical(f"Skipping, segment StartTime or endTime ({startTime},{endTime}) longer than wav file ({audio_info.duration})")
                        continue
                else:
                    # Text only
                    startTime = 0.0
                    endTime = 1.0
                    wav_path = "/dev/null"

                if split == "train":
                    train_transcript_words.append(text)
                    for word in set(text.split(" ")):
                        train_vocab.add(word)

                info.append({
                    "ID" : f"{parts[0]}-{int(float(parts[3])*100):07d}-{int(float(parts[4])*100):07d}",
                    "text" : text,
                    "spk" : parts[2],
                    "gender" : f"{parts[5].split(',', 3)[2].replace('>', '')}",
                    "startTime": startTime,
                    "endTime": endTime,
                    "duration": endTime - startTime,
                    "file": wav_path,
                })


    for _, split, info in split_info:

        # Sort the data based on column ID
        sorted_formatted_data = sorted(info, key=lambda x: x["ID"])

        if len(sorted_formatted_data) == 0:
            logger.critical(f"No file found for {info}, check directory paths to the datasets")
            sys.exit(1)

        csv_file = os.path.join(save_folder, split + ".csv")
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted_formatted_data[0].keys())
            writer.writeheader()
            writer.writerows(sorted_formatted_data)

    sorted_vocabulary = sorted(train_vocab)
    vocab_file = os.path.join(save_folder, "vocab.txt")
    with open(vocab_file, "w") as file:
        for word in sorted_vocabulary:
            if word == " ":
                continue
            if word == "":
                continue
            # Write each word to the file, followed by a newline character
            file.write(word + "\n")

    transcript_words = os.path.join(save_folder, "transcript_words.txt")
    with open(transcript_words, "w") as file:
        for line in train_transcript_words:
            file.write(line + "\n")

    # saving options
    save_pkl(conf, save_opt)

def normalize_wav_key(key):
    key = key.replace("suite", "")
    key = key.replace("_bis", "")
    key = key.replace("automatique", "")
    key = key.replace("wav", "")
    key = key.replace("-","_")
    key = key.replace(".","")
    key = key.lower()
    return key


def custom_glob_filter(directory):
    # Support for exclude exact word
    # https://stackoverflow.com/questions/20638040/glob-exclude-pattern
    try:  # Try to parse exact match direction
        exclude_match = re.findall(r"\[\^.*\]", directory)[0].replace('[^', '').replace(']', '')
    except IndexError:
        exclude_match = None
    else:  # Remove custom directive
        directory = re.sub(r"\[\^.*\]", "", directory)
    paths = glob.glob(directory, recursive = True)
    return paths, exclude_match

def transform_lines(line):
    # Perform string replacements using regular expressions
    line = [re.sub(r'<F0_M>', '<o,f0,male>', line) for line in line]
    line = [re.sub(r'<F0_F>', '<o,f0,female>', line) for line in line]
    line = [re.sub(r'\([0-9]+\)', '', line) for line in line]
    line = [re.sub(r'<sil>', '', line) for line in line]
    line = [re.sub(r'\([^ ]*\)$', '', line) for line in line]
    return line

def text_transform(text):

    for target, replacement in broken_encoding_replacements.items():
        text = text.replace(target, replacement)


    # Names
    text = re.sub(r"Franç§ois", "François", text)
    text = re.sub(r"Schrà ¶der", "Schràder", text)

    text = re.sub(r"«", "", text)
    text = re.sub(r"»", "", text)

    text = re.sub(r"°", "degré", text)

    text = re.sub(r"²", "", text)

    # remove html tag
    text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', "", text)

    # Replace curly braces with square brackets
    text = text.replace('{', '[').replace('}', ']')

    text = re.sub(r"\.\.\.|\*|\[.*?\]", "", text.lower())
    delset_specific = delset
    remove_clear = "()=-"
    for char in remove_clear:
        delset_specific = delset_specific.replace(char, "")
    text = text.translate(str.maketrans("", "", delset_specific))

    # Undecidable variant heared like on (n') en:
    text = re.sub(r"\(.+?\)", "", text)
    text = re.sub(r"\(\)", "", text)
    text = re.sub(r"(O.K.)", "ok", text)
    text = re.sub(r"(O.K)", "ok", text)

    text = re.sub(r"%", "pourcent", text)

    text = re.sub(r"=", "", text)
    text = re.sub(r"\(", "", text)
    text = re.sub(r"\)", "", text)

    # t 'avais
    text = re.sub(r"[ ]\'", " ", text)
    text = re.sub(r"\'", "' ", text)

    # ' en debut de phrase
    text = re.sub(r"^'", "", text)

    # -) hesitation
    text = re.sub(r"-\)", "", text)

    num_list = re.findall(" \d+,\d+ | \d+,\d+$", text)
    if len(num_list) > 0:
        for num in num_list:
            num_in_word = num2words(float(num.replace(",", ".")), lang="fr")
            text = text.replace(num, " " + str(num_in_word) + " ", 1)

    num_list = re.findall("\d+,\d+", text)
    if len(num_list) > 0:
        for num in num_list:
            num_in_word = num2words(float(num.replace(",", ".")), lang="fr")
            text = text.replace(num, " " + str(num_in_word) + " ", 1)

    num_list = re.findall(" \d+ | \d+$", text)
    if len(num_list) > 0:
        for num in num_list:
            num_in_word = num2words(int(num), lang="fr")
            text = text.replace(num, " " + str(num_in_word) + " ", 1)

    num_list = re.findall("\d+", text)
    if len(num_list) > 0:
        for num in num_list:
            num_in_word = num2words(int(num), lang="fr")
            text = text.replace(num, " " + str(num_in_word) + " ", 1)

    # arc-en-ciel
    text = re.sub(r"-", " ", text)

    # virgule (after num2words!)
    text = re.sub(r",", "", text)

    # euh
    # text = re.sub(r"euh", "", text)

    # ã used as à in most case
    text = re.sub(r"ã", "à", text)

    # replace n succesive spaces with one space.
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub("^ ", "", text)
    text = re.sub(" $", "", text)

    # The byte 0x9c encodes a "curly quote" in the Windows-1252 character encoding.
    text = re.sub(r"c½ur", "coeur", text)
    text = re.sub(r"cur", "coeur", text)
    # The byte 0x92 encodes a "curly quote" in the Windows-1252 character encoding.
    text = re.sub(r"", "'", text)
    text = re.sub(r"' '", "' ", text)
    text = re.sub(r"'' ", "' ", text)

    return text


def filterout_lines(lines):
    # Filter out lines containing specific patterns
    return [
        line
        for line in lines
        if not any(pattern in line for pattern in ['ignore_time_segment_in_scoring', ';;', 'inter_segment_gap', 'excluded_region'])
    ]


def skip(save_folder, conf, save_opt):
    """
    Detect when data prep can be skipped.

    Arguments
    ---------
    save_folder : str
        The location of the seave directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    if len(glob.glob(os.path.join(save_folder, "*.csv"), recursive = False)) == 0:
        logger.info(f"Did not found any csv in '{save_folder}'")
        skip = False
    else:
        logger.info(f"Found csv in '{save_folder}'")

    #  Checking saved options
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip
