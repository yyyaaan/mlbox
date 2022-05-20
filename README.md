# Machine Lerning Box

A few options of transfer learning and use of Azure Machine Learning Studio

`tensorflow` on Apple M1 may be different from other platform.

Use respective conda environments to smooth web deployment. 

```
conda info --envs
conda activate ~/miniforge3/envs/tfcv
conda deactivate
```

## Azure ML Studio notes

[AzureML examples](https://github.com/Azure/azureml-examples)

> Easy tensorflow tracking to ML studio using [custom callback] (https://towardsdatascience.com/logging-tensorflow-keras-metrics-to-azure-ml-studio-in-realtime-14504a01cad8) 

> Save computation cost by using local compute: [traning on different compute target](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/training)


## Docker

docker build . -t tfenv

Docker -ps -a
Docker start - name (bg-run instead of “run”, which create a new one)

Docker run -it , start -ai (interactive and terminal, start attach)

 —rm remove automatically on exit   

## Playground: Husky names 

Training using `tflite-model-maker` on [Colab](https://colab.research.google.com/drive/168oDoHFb6g5LtGlDau4CUy0UOzTqh8q8)