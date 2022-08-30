import pandas
import os

# wav_dir_path = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/recaptcha5k/val"
wav_dir_path = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/recaptcha5k/train"
wavs = os.listdir(wav_dir_path)
files = []
# csv_path = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/recaptcha5k_csv"
csv_path = "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/recaptcha5k_csv"
# _csv = os.listdir(csv_path)

trans = "custom target phrase"
for wavlist in wavs:
    # trans = wavlist.split(".")[0]
    wav_path = os.path.join(wav_dir_path, wavlist)
    wav_size = os.path.getsize(wav_path)
    files.append((os.path.abspath(wav_path), wav_size, trans))

dataset = pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])
dataset.to_csv(os.path.join(csv_path, "recaptcha5k_train.csv"), index = False)
