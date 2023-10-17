#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer.
The tokenizer converts words into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).
When doing a speech recognition experiment you have to make
sure that the acoustic and language models are trained with
the same tokenizer. Otherwise, a token mismatch is introduced
and beamsearch will produce bad results when combining AM and LM.

To run this recipe, do the following:
> python train.py hyperparams/some.yaml


Authors
 * Abdel Heba 2021
"""

import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset prep (using glob pattern matching from data_folder)
    from stm_prepare import prepare_stm  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_stm,
        kwargs={
            "stm_directory": hparams["stm_directory"],
            "wav_directory": None,
            "tr_splits": hparams["tr_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["te_splits"],
            "save_folder": hparams["prep_save_folder"],
            "skip_prep": hparams["skip_prep"],
            "ignore_wav": True,
        },
    )

    # Train tokenizer
    hparams["tokenizer"]()
