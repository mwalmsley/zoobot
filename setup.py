import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zoobot",
    version="1.0.5",
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
            'torch == 1.12.1+cpu',
            'torchvision == 0.13.1+cpu',
            'torchaudio == 0.12.1',
            'pytorch-lightning >= 2.0.0',
            # 'simplejpeg',
            'albumentations',
            'pyro-ppl == 1.8.0',
            'torchmetrics == 0.11.0',
            'timm == 0.6.12'
        ],
        'pytorch_m1': [
            # as above but without the +cpu (and the extra-index-url in readme has no effect)
            # all matching pytorch versions for an m1 system will be cpu
            'torch == 1.12.1',
            'torchvision == 0.13.1',
            'torchaudio == 0.12.1',
            'pytorch-lightning >= 2.0.0',
            'albumentations',
            'pyro-ppl == 1.8.0',
            'torchmetrics == 0.11.0',
            'timm == 0.6.12'
        ],
        # as above but without pytorch itself
        # for GPU, you will also need e.g. cudatoolkit=11.3, 11.6
        # https://pytorch.org/get-started/previous-versions/#v1121
        'pytorch_cu113': [
            'torch == 1.12.1+cu113',
            'torchvision == 0.13.1+cu113',
            'torchaudio == 0.12.1',
            'pytorch-lightning >= 2.0.0',
            'albumentations',
            'pyro-ppl == 1.8.0',
            'torchmetrics == 0.11.0',
            'timm == 0.6.12'
        ],
        'pytorch_colab': [
            # colab includes pytorch already
            'pytorch-lightning >= 2.0.0',
            'albumentations',
            'pyro-ppl>=1.8.0',
            'torchmetrics==0.11.0',
            'timm == 0.6.12'
        ],
        # TODO may add narval/Digital Research Canada config
        'tensorflow': [
            'tensorflow == 2.10.0',  # 2.11.0 turns on XLA somewhere which then fails on multi-GPU...TODO
            'keras_applications',
            'tensorflow_probability == 0.18.0',  # 0.19 requires tf 2.11
            'protobuf <= 3.19'  # tensorflow incompatible above this (usually resolved by pip automatically)
        ],
        # for GPU, you will also need cudatoolkit=11.2 and cudnn=8.1.0 (note - 11.3 NOT supported by TF)
        # https://www.tensorflow.org/install/pip#step-by-step_instructions
        'utilities': [
            'seaborn',  # for nice plots
            'boto3',    # for AWs s3 access
            'python-dateutil == 2.8.1',  # for boto3  
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
        'astropy',  # for reading fits
        'scikit-learn >= 1.0.2',
        'matplotlib',
        'pyarrow',  # to read parquet, which is very handy for big datasets
        # for saving metrics to weights&biases (cloud service, free within limits)
        'wandb',
        'setuptools',  # no longer pinned
        'galaxy-datasets>=0.0.15'  # for dataset loading in both TF and Torch (see github/mwalmsley/galaxy-datasets)
    ]
)
