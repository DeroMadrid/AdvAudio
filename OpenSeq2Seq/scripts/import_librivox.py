#!/usr/bin/env python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2018 Mozilla Corporation


from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import codecs
import fnmatch
import pandas
import tqdm
import subprocess
import tarfile
import unicodedata

from sox import Transformer
import urllib
from tensorflow.python.platform import gfile

def _maybe_download(fname, data_dir, data_url):
  data_path = os.path.join(data_dir, fname)
  if not os.path.exists(data_path):
    print("Can't find '{}'. Downloading...".format(data_path))
    urllib.request.urlretrieve(data_url, filename=data_path + '.tmp')
    os.rename(data_path + '.tmp', data_path)
  else:
    print("Skipping file '{}'".format(data_path))
  return data_path

def _download_and_preprocess_data(data_dir):
  # Conditionally download data to data_dir
  print("Downloading Librivox data set (55GB) into {} if not already present...".format(data_dir))
  with tqdm.tqdm(total=7) as bar:
    TRAIN_CLEAN_100_URL = "https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz"
    TRAIN_CLEAN_360_URL = "https://openslr.magicdatatech.com/resources/12/train-clean-360.tar.gz"
    TRAIN_OTHER_500_URL = "https://openslr.magicdatatech.com/resources/12/train-other-500.tar.gz"

    DEV_CLEAN_URL = "https://openslr.magicdatatech.com/resources/12/dev-clean.tar.gz"
    DEV_OTHER_URL = "https://openslr.magicdatatech.com/resources/12/dev-other.tar.gz"

    TEST_CLEAN_URL = "https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz"
    TEST_OTHER_URL = "https://openslr.magicdatatech.com/resources/12/test-other.tar.gz"

    def filename_of(x): return os.path.split(x)[1]
    train_clean_100 = _maybe_download(filename_of(TRAIN_CLEAN_100_URL), data_dir, TRAIN_CLEAN_100_URL)
    bar.update(1)
    train_clean_360 = _maybe_download(filename_of(TRAIN_CLEAN_360_URL), data_dir, TRAIN_CLEAN_360_URL)
    bar.update(1)
    train_other_500 = _maybe_download(filename_of(TRAIN_OTHER_500_URL), data_dir, TRAIN_OTHER_500_URL)
    bar.update(1)

    dev_clean = _maybe_download(filename_of(DEV_CLEAN_URL), data_dir, DEV_CLEAN_URL)
    bar.update(1)
    dev_other = _maybe_download(filename_of(DEV_OTHER_URL), data_dir, DEV_OTHER_URL)
    bar.update(1)

    test_clean = _maybe_download(filename_of(TEST_CLEAN_URL), data_dir, TEST_CLEAN_URL)
    bar.update(1)
    test_other = _maybe_download(filename_of(TEST_OTHER_URL), data_dir, TEST_OTHER_URL)
    bar.update(1)

  # Conditionally extract LibriSpeech data
  # We extract each archive into data_dir, but test for existence in
  # data_dir/LibriSpeech because the archives share that root.
  print("Extracting librivox data if not already extracted...")
  with tqdm.tqdm(total=7) as bar:
    LIBRIVOX_DIR = "LibriSpeech"
    work_dir = os.path.join(data_dir, LIBRIVOX_DIR)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-100"), train_clean_100)
    bar.update(1)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-360"), train_clean_360)
    bar.update(1)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-other-500"), train_other_500)
    bar.update(1)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
    bar.update(1)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-other"), dev_other)
    bar.update(1)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
    bar.update(1)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)
    bar.update(1)

  # Convert FLAC data to wav, from:
  # data_dir/LibriSpeech/split/1/2/1-2-3.flac
  # to:
  # data_dir/LibriSpeech/split-wav/1-2-3.wav
  #
  # And split LibriSpeech transcriptions, from:
  # data_dir/LibriSpeech/split/1/2/1-2.trans.txt
  # to:
  # data_dir/LibriSpeech/split-wav/1-2-0.txt
  # data_dir/LibriSpeech/split-wav/1-2-1.txt
  # data_dir/LibriSpeech/split-wav/1-2-2.txt
  # ...
  print("Converting FLAC to WAV and splitting transcriptions...")
  with tqdm.tqdm(total=7) as bar:
    train_100 = _convert_audio_and_split_sentences(work_dir, "train-clean-100", "train-clean-100-wav")
    bar.update(1)
    train_360 = _convert_audio_and_split_sentences(work_dir, "train-clean-360", "train-clean-360-wav")
    bar.update(1)
    train_500 = _convert_audio_and_split_sentences(work_dir, "train-other-500", "train-other-500-wav")
    bar.update(1)

    dev_clean = _convert_audio_and_split_sentences(work_dir, "dev-clean", "dev-clean-wav")
    bar.update(1)
    dev_other = _convert_audio_and_split_sentences(work_dir, "dev-other", "dev-other-wav")
    bar.update(1)

    test_clean = _convert_audio_and_split_sentences(work_dir, "test-clean", "test-clean-wav")
    bar.update(1)
    test_other = _convert_audio_and_split_sentences(work_dir, "test-other", "test-other-wav")
    bar.update(1)

  # Write sets to disk as CSV files
  train_100.to_csv(os.path.join(data_dir, "librivox-train-clean-100.csv"), index=False)
  train_360.to_csv(os.path.join(data_dir, "librivox-train-clean-360.csv"), index=False)
  train_500.to_csv(os.path.join(data_dir, "librivox-train-other-500.csv"), index=False)

  dev_clean.to_csv(os.path.join(data_dir, "librivox-dev-clean.csv"), index=False)
  dev_other.to_csv(os.path.join(data_dir, "librivox-dev-other.csv"), index=False)

  test_clean.to_csv(os.path.join(data_dir, "librivox-test-clean.csv"), index=False)
  test_other.to_csv(os.path.join(data_dir, "librivox-test-other.csv"), index=False)

def _maybe_extract(data_dir, extracted_data, archive):
  # If data_dir/extracted_data does not exist, extract archive in data_dir
  if not gfile.Exists(os.path.join(data_dir, extracted_data)):
    tar = tarfile.open(archive)
    tar.extractall(data_dir)
    tar.close()

def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir):
  source_dir = os.path.join(extracted_dir, data_set)
  target_dir = os.path.join(extracted_dir, dest_dir)

  if not os.path.exists(target_dir):
    os.makedirs(target_dir)

  # Loop over transcription files and split each one
  #
  # The format for each file 1-2.trans.txt is:
  # 1-2-0 transcription of 1-2-0.flac
  # 1-2-1 transcription of 1-2-1.flac
  # ...
  #
  # Each file is then split into several files:
  # 1-2-0.txt (contains transcription of 1-2-0.flac)
  # 1-2-1.txt (contains transcription of 1-2-1.flac)
  # ...
  #
  # We also convert the corresponding FLACs to WAV in the same pass
  files = []
  for root, dirnames, filenames in os.walk(source_dir):
    for filename in fnmatch.filter(filenames, '*.trans.txt'):
      trans_filename = os.path.join(root, filename)
      with codecs.open(trans_filename, "r", "utf-8") as fin:
        for line in fin:
          # Parse each segment line
          first_space = line.find(" ")
          seqid, transcript = line[:first_space], line[first_space+1:]

          # We need to do the encode-decode dance here because encode
          # returns a bytes() object on Python 3, and text_to_char_array
          # expects a string.
          transcript = unicodedata.normalize("NFKD", transcript) \
                      .encode("ascii", "ignore")   \
                      .decode("ascii", "ignore")

          transcript = transcript.lower().strip()

          # Convert corresponding FLAC to a WAV
          flac_file = os.path.join(root, seqid + ".flac")
          wav_file = os.path.join(target_dir, seqid + ".wav")
          if not os.path.exists(wav_file):
            Transformer().build(flac_file, wav_file)
          wav_filesize = os.path.getsize(wav_file)

          files.append((os.path.abspath(wav_file), wav_filesize, transcript))

  return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

if __name__ == "__main__":
  _download_and_preprocess_data(sys.argv[1])
