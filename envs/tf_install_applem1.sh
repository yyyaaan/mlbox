# require Miniforge installed
# https://developer.apple.com/metal/tensorflow-plugin/

source ~/miniforge3/bin/activate

conda install -c apple tensorflow-deps
conda install openblas

python -m pip install -U pip
python -m pip install tensorflow-macos tensorflow-metal
pip install numpy opencv-python pillow tflite-support
pip install matplotlib azureml-sdk azureml-widgets

# python -m pip install tflite-model-maker-nightly


