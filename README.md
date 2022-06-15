# Machine Learning Box

A few options of transfer learning and use of Azure Machine Learning Studio

`tensorflow` on Apple M1 may be different from other platform.

Use respective conda environments to smooth web deployment. 

```
conda info --envs
conda activate ~/miniforge3/envs/tfcv
conda deactivate
conda env create -f tfmtcnn.yaml
```

## Hot topics

[Image segementation](https://www.tensorflow.org/tutorials/images/segmentation)

[Imagen Text-to-Pic](https://imagen.research.google/)


## Azure ML Studio notes

[AzureML examples](https://github.com/Azure/azureml-examples)

> Easy tensorflow tracking to ML studio using [custom callback] (https://towardsdatascience.com/logging-tensorflow-keras-metrics-to-azure-ml-studio-in-realtime-14504a01cad8) 

> Save computation cost by using local compute: [training on different compute target](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/training)


## Docker

docker build . -t tfenv

Docker -ps -a
Docker start - name (bg-run instead of “run”, which create a new one)

Docker run -it , start -ai (interactive and terminal, start attach)

 —rm remove automatically on exit   

## Playground: Husky names 

Training using `tflite-model-maker` on [Colab](https://colab.research.google.com/drive/168oDoHFb6g5LtGlDau4CUy0UOzTqh8q8)