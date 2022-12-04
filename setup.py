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
    python_requires=">=3.9",  # handy for logging.basicConfig(force=True). 3.10 TBD
    extras_require={
        'pytorch': [
            'torch == 1.10.1',
            'torchvision == 0.11.2',
            'torchaudio == 0.10.1',
            'pytorch-lightning==1.6.5',  # 1.7 requires protobuf version incompatible with tensorflow/tensorboard. Otherwise works.
            'simplejpeg',
            'albumentations',
            'pyro-ppl == 1.8.0'
        ],
        'tensorflow': [
            'tensorflow >= 2.11.0',
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
        'galaxy-datasets==0.0.2'  # for dataset loading in both TF and Torch (renamed from pytorch-galaxy-datasets)
    ]
)
