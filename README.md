# MXNet on the iPhone

This is an example project running a binarized neural network on iOS. It can classify the live camera feed.

## Requirements

- ios device
- model (binarized version pre-trained on Imagenet included)

## Usage

Use xcode to ``build`` the app and try it on your phone!

## What it does

Under the hood, we need the amalgamated MXNet source, the c predict headers, a trained and saved model from mxnet and set some additional environment variables in xcode.

The mxnet script ``amalgamate_mxnet.sh`` will amalgamate mxnet and perform the changes necessary for ios as described in the amalgamation readme file.

The xcode project already contains the preprocessor settings required to build mxnet for ios:
- ``"MXNET_PREDICT_ONLY=1"``
- ``"MXNET_USE_OPENCV=0"``
- ``"MSHADOW_USE_CUDA=0"``
- ``"MSHADOW_USE_SSE=0"``
- ``"BINARY_WORD_32=1"`` (set to 32bit for ARM7 devices)
- ``"BINARY_WORD_64=0"``

There are pre-trained models included in the projects.
