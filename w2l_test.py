# pylint: skip-file
import tensorflow as tf
from OpenSeq2Seq.open_seq2seq.models import Speech2Text
from OpenSeq2Seq.open_seq2seq.encoders import TDNNEncoder
from OpenSeq2Seq.open_seq2seq.decoders import FullyConnectedCTCDecoder
from OpenSeq2Seq.open_seq2seq.data import Speech2TextDataLayer
from OpenSeq2Seq.open_seq2seq.losses import CTCLoss
from OpenSeq2Seq.open_seq2seq.optimizers.lr_policies import poly_decay


base_model = Speech2Text

base_params = {
    "random_seed": 0,
    "use_horovod": False,
    "num_epochs": 20,

    "num_gpus": 1,
    "batch_size_per_gpu": 8,
    "iter_size": 1,

    "save_summaries_steps": 100,
    "print_loss_steps": 10,
    "print_samples_steps": 200,
    "eval_steps": 200,
    "save_checkpoint_steps": 1000,
    # "num_checkpoints": 1,
    "logdir": "w2lplus_model",

    "optimizer": "Momentum",
    "optimizer_params": {
        "momentum": 0.90,
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 0.05,
        "power": 2.0,
    },
    "larc_params": {
        "larc_eta": 0.001,
    },

    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.001
    },

    # "dtype": tf.float32,
    "dtype": "mixed",
    "loss_scaling": "Backoff",

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": TDNNEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [29], "stride": [1],
                "num_channels": 896, "padding": "SAME",
                "dilation":[2], "dropout_keep_prob": 0.6,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [1], "stride": [1],
                "num_channels": 1024, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.6,
            }
        ],

        "dropout_keep_prob": 0.7,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_last",
    },

    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "initializer": tf.contrib.layers.xavier_initializer,
        "use_language_model": False,

        # params for decoding the sequence with language model
        "beam_width": 512,
        "alpha": 2.0,
        "beta": 1.5,

        "decoder_library_path": "OpenSeq2Seq/ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
        "lm_path": "OpenSeq2Seq/language_model/4-gram.binary",
        "trie_path": "OpenSeq2Seq/language_model/trie.binary",
        "alphabet_config_path": "OpenSeq2Seq/open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    },
    "loss": CTCLoss,
    "loss_params": {},
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "OpenSeq2Seq/open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "data/librispeech/librivox-train-clean-100.csv",
        ],
        "max_duration": 16.7,
        "shuffle": True,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "OpenSeq2Seq/open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            # "data/librispeech/librivox-dev-clean.csv",
            "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/recaptcha5k_csv/recaptcha5k_val.csv",
        ],
        "shuffle": False,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "OpenSeq2Seq/open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            # "data/librispeech/librivox-test-clean.csv",
            "/media/ps/data/gxy/Adv_audio/data/recaptchaV2/recaptcha5k_csv/recaptcha5k_val.csv",
        ],
        "shuffle": False,
    },
}