# Zoobot batch processing API developer readme

## Build the Docker image for Azure Batch node pools

We need to build a Docker image with the necessary packages (ML system Pytorch or TensorFlow) to run the code in the Azure Batch ecosystem.

Azure Batch will pull this image from a private container registry, which needs to be in the same region as the Batch account.

Build the image from the Dockerfile in this folder:

``` sh
export IMAGE_NAME=zoobot.azurecr.io/pytorch:1.10.1-gpu-py3
export REGISTRY_NAME=zoobot
docker-compose build
docker image tag zoobot:cuda $IMAGE_NAME
```

Test that TensorFlow can use the GPU in an interactive Python session:

``` sh
docker run --gpus all -it --rm $IMAGE_NAME /bin/bash

python
>>> import torch
>>> print(torch.cuda.is_available())
>>> print(torch.cuda.device_count())
>>> print(torch.cuda.current_device())
>>> print(torch.cuda.get_device_name(0))
>>> quit()
```

You can now exit/stop the container.

Log in to the Azure Container Registry for the batch API project and push the image; you may have to `az login` first:

``` sh
az acr login --name $REGISTRY_NAME

docker image push $IMAGE_NAME
```

## Create a Azure Batch node pool

We create a separate node pool for each instance of a deployed Zoobot image. For example, our default `training` instance of Zoobot has one node pool.

Follow the `examples/create_batch_pool.ipynb` notebook in the PR at [create_batch_pool PR](https://github.com/zooniverse/panoptes-python-notebook/pull/4) to create one. You should only need to do this for new instances of Zoobot.
