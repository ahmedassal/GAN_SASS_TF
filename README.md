# GAN_SASS_TF
TensorFlow implementation of "GAN Single Audio Source Separation"

## WORK IN PROGRESS

The model currently incomplete, having performance issues, and doesn't perform the task well ... yet.

## Dependencies

[python]
- six
- numpy, scipy, nltk
- tensorflow 1.12+

[TIMIT dataset]
- install "sndfile-convert" utility
    `sudo apt-get install sndfile-programs`

[optional]
- install warp-ctc for better GPU CTC performance
    `git clone https://github/baidu-research/warp-ctc`
    then follow instructions inside to install

## Prepare datasets

### toy dataset

There is a "toy dataset" for debugging purposes. It's just pure generated white noise.
In `app/hparams.py` file, set `DATASET_TYPE = 'toy'` to use this dataset.

### TIMIT

 - Make sure to install `sndfile-tools` first (for `sndfile-convert` utility)
 - Download the `TIMIT.zip` file under `app/datasets/TIMIT/`.
 - Under repo directory `app/datasets/TIMIT/`, run `install.sh`. You only need to run it for once.
 - If you change hyperparameter `FFT_SIZE` or `CHARSET_SIZE`, you need to re-run install script.
 - In `app/hparams.py` file, set `DATASET = 'timit'` to use this dataset

## Quick Instructions

To quickly train the model, modify `app/hparams.py` file to setup hyperparameter,
then run `python main.py -o saves/my_model.cpkt`.

This will will save model parameters in `saves/my_model.cpkt`

To continue training, run `python main.py -i saves/my_model.cpkt`

To check model accuracy on test set, do `python main.py -i <YOUR_MODEL_PARAMETERS> -m=test`

To quickly use the model to separate a WAV file, do `python main.py -i <YOUR_MODEL_PARAMETERS> -m=demo -if=your_audio.wav`.

For more CLI arguments help, do `python main.py --help`

### Code structure

`main.py` defines global model architecture, training / testing, and basic CLI funcitonality

`app/datasets` contains code to preprocess / access datasets.

`app/hparams.py` defines hyperparameters for quick finetune.

**NOTE** ASR module is currently buggy.

This defines common hyperparameters such as learn rate, batch size. As well as submodule types to use.

`app/modules.py` defines architecture for sub-modules that are used in the model, such as separator, discriminator.
You can add your own sub-modules here. For example:

```
@register_separator('my_separator')
class MySeparator(Separator):
    ...
```

Then change `hparams.py`:

```
SEPARATOR_TYPE = 'my_separator'
```

`app/ops.py` defines some commonly used Tensorflow Ops.

`app/ozers.py` defines optimizers.

`app/utils.py` defines commonly used subroutines.

`saves/` directory intends to store model parameters.
Run `clear_saves.sh` to remove everything there. You can put them elsewhere though.

`logs/` directory stores TF summaries. Start TensorBoard via `tensorboard --logdir=logs/` to visualize them.
`asr_logs/` directory stores TF summaries for speech recgonizer.

### Coding style

This codebase mostly follow PEP8, but there are a few exceptions.

For symbolics:
    - `s_var` means a placeholder or **s**ymbolic element in a graph.
    - `sS_var` means it's also **s**parse.
    - `v_var` means a tensor variable with **v**alue, typically parameter.
