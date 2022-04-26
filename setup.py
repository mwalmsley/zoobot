import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zoobot",
    version="0.0.3",
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
    python_requires=">=3.7",  # tf 2.8.0 requires Python 3.7 and above
    extras_require={
        'pytorch': [
            'torch == 1.10.1',
            'torchvision == 0.11.2',
            'torchaudio == 0.10.1',
            'pytorch-lightning',
            'simplejpeg',
            'albumentations',
            'pyro-ppl == 1.8.0'
        ],
        'tensorflow': [
            'tensorflow >= 2.8',
            'keras_applications',
            'tensorflow_probability >= 0.11'
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
        'python-dateutil == 2.8.1',  # for boto3
        'boto3',
        'statsmodels',
        'pyarrow',  # to read parquet, which is very handy for big datasets
        'seaborn',  # for nice plots
        #'astropy', # for reading .fits (not yet implemented, but likely to be added)
        # optional, for saving metrics to weights&biases (cloud service, free within limits)
        'wandb'
    ]
)
