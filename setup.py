import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zoobot",
    version="0.0.4",
    author="Mike Walmsley",
    author_email="walmsleymk1@gmail.com",
    description="Galaxy morphology classifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwalmsley/zoobot",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",  # recommend 3.9 for new users. TF needs >=3.7.2, torchvision>=3.8
    extras_require={
        'pytorch_cpu': [
            # A100 GPU currently only seems to support cuda 11.3 on manchester cluster, let's stick with this version for now
            # very latest version wants cuda 11.6
            'torch == 1.12.1',
            'torchvision == 0.13.2',
            'torchaudio == 0.12.1',
            'pytorch-lightning==1.8.3.post1',  # tensorboard/protobuf issue fixed now
            'simplejpeg',
            'albumentations',
            'pyro-ppl == 1.8.0',
            'torchmetrics == 0.11.0'
        ],
        # as above but without pytorch itself
        # for GPU, you will also need e.g. cudatoolkit=11.3, 11.6
        # https://pytorch.org/get-started/previous-versions/#v1121
        'pytorch_gpu': [
            'pytorch-lightning==1.8.3.post1',
            'simplejpeg',
            'albumentations',
            'pyro-ppl == 1.8.0',
            'torchmetrics == 0.11.0'
        ],
        'tensorflow_cpu': [
            'tensorflow == 2.10.0',  # 2.11.0 turns on XLA somewhere which then fails on multi-GPU...TODO
            'keras_applications',
            'tensorflow_probability >= 0.18.0',
            'protobuf <= 3.19'  # tensorflow incompatible above this (usually resolved by pip automatically)
        ],
        # as above but without TensorFlow itself
        # for GPU, you will also need cudatoolkit=11.2 and cudnn=8.1.0 (note - 11.3 NOT supported by TF)
        # https://www.tensorflow.org/install/pip#step-by-step_instructions
        'tensorflow_gpu': [
            # 'tensorflow == 2.10.0',  # actually can install tensorflow like this but let's be consistent with Torch
            'keras_applications',
            'tensorflow_probability >= 0.18.0',
            'protobuf <= 3.19'  # tensorflow incompatible above this (usually resolved by pip automatically)
        ],
        'utilities': [
            'seaborn',  # for nice plots
            'boto3',    # for AWs s3 access
            'python-dateutil == 2.8.1',  # for boto3
            'astropy' # for reading .fits (not yet implemented, but likely to be added)
        ],
        'docs': [
            'Sphinx',
            'sphinxcontrib-napoleon',
            'furo',
            'docutils<0.18'
        ]
    },
    install_requires=[
        'h5py',
        'tqdm',
        'pillow',
        'numpy',
        'pandas',
        'scipy',
        'scikit-image >= 0.19.2',
        'scikit-learn >= 1.0.2',
        'matplotlib',
        'pyarrow',  # to read parquet, which is very handy for big datasets
        'statsmodels',
        # for saving metrics to weights&biases (cloud service, free within limits)
        'wandb',
        'setuptools==59.5.0',  # wandb logger incompatibility
        'galaxy-datasets==0.0.3'  # for dataset loading in both TF and Torch (renamed from pytorch-galaxy-datasets)
    ]
)
