---
title: 'Zoobot: Adaptable Deep Learning Models for Galaxy Morphology'
tags:
  - Python
  - astronomy
  - deep learning
  - galaxy morphology
  - statistics
  - citizen science
authors:
  - name: Mike Walmsley
    orcid: 0000-0002-6408-4181
    corresponding: true
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Cambell Allen
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Micah Bowles
    corresponding: false
    affiliation: 1
  - name: Inigo Val
    corresponding: false
    affiliation: 1
  - name: Devina Mohan
    corresponding: false
    affiliation: 1
  - name: Ben
    corresponding: false
    affiliation: 1
  - name: Chris J. Lintott
    corresponding: false
    affiliation: 1
  - name: Anna M. M. Scaife
    corresponding: false
    affiliation: 1
  - name: Kasia
    corresponding: false
    affiliation: TODO
affiliations:
 - name: Department of Physics and Astronomy, University of Manchester, Manchester, UK
   index: 1
 - name: Zooniverse.org, University of Oxford, Oxford, UK
   index: 2
 - name: MPIA
   index: 3
date: 19 February 2023
bibliography: paper.bib

---

# Summary

<!--  Summary: Has a clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided? -->

`Zoobot` is a Python package for measuring the detailed appearance of galaxies in telescope images
using deep learning.
`Zoobot` is for astronomers who want to solve a galaxy image task (e.g. finding merging galaxies, counting spiral arms).
Astronomers can use `Zoobot` to adapt (finetune) pretrained deep learning models to solve their task.
These finetuned models perform better and require far fewer new labels than training from scratch `[@Walmsley2022Towards]`.

The models included with `Zoobot` are pretrained on up to 92 million responses from [Galaxy Zoo](www.galaxyzoo.org) volunteers.
Each volunteer answers a series of tasks describing the detailed appearance of each galaxy. 
`Zoobot`'s models are trained to answer all of these tasks simultaneously.
Having already learned to solve those tasks, the models can then be adapted to new related tasks.

`Zoobot` provides a high-level API and guided workflow for carrying out the finetuning process.
The API abstracts away engineering details such as efficiently loading astronomical images, multi-GPU training, iteratively finetuning deeper model layers, etc.
Behind the scenes, these steps are implemented via either PyTorch [@Pytorch2019] or TensorFlow [@https://doi.org/10.48550/arxiv.1603.04467], according to the user's choice.
`Zoobot` is therefore accessible to astronomers with no previous experience in deep learning.

For advanced users, `Zoobot` also includes the code to replicate and extend `Zoobot`'s pretrained models.
This is used routinely at [Galaxy Zoo](www.galaxyzoo.org) to scale up galaxy measurement catalogues `[@Walmsley2022decals]`
and to prioritise the galaxies shown to volunteers for labelling.
Zoobot models have been applied to measure galaxy appearance in SDSS `[@Walmsley:2021]`, Hubble, HSC, and DESI, and are included in the data pipeline of upcoming space telescope Euclid `[@2011arXiv1110.3193L]`.
We hope that `Zoobot` will help astronomers use deep learning and the next generation of survey telescopes to answer their own science questions.

# Statement of need
<!-- A statement of need: Does the paper have a section titled ‘Statement of need’ that clearly states what problems the software is designed to solve, who the target audience is, and its relation to other work? -->

Astronomers aim to understand why galaxies look the way they do by measuring
the appearance - morphology - of millions of galaxies `[@Masters2019a]`
The sheer number of images requires most of these measurement to be made automatically with software `[@Walmsley:2021]`.

Unfortunately, making automated measurements of complicated features like spiral arms is difficult because
it is hard to write down a set of steps that reliably identify those and only those features.
This mirrors many image classification problems back on Earth `[@LeCun2015]`.
Astronomers often aim instead to learn the measurement steps directly from data
by providing deep learning models with large sets of images with labels (e.g. spiral or not) `[@HuertasCompany2022]`.

Gathering large sets of labelled galaxy images is a major barrier.
Models trained on millions to billions of labelled images consistently perform better [@Bommasani2021;@https://doi.org/10.48550/arxiv.2302.05442].
Astronomers cannot routinely label this many images.
Neither can most other people;
terrestrial practictioners often start with a model already trained ("pretrained")
on a broad generic task and then adapt it ("finetune") to their specific measurement task `[@https://doi.org/10.48550/arxiv.2104.10972]`.

Zoobot by sharing models pre-trained
Further, Zoobot 

new surveys

<!-- State of the field: Do the authors describe how this software compares to other commonly-used packages? -->

<!-- - `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->


# Acknowledgements

Zoobot was made possible by the Galaxy Zoo volunteers,
who collectively created the crucial morphology labels used to create our pretrained models (and much, much more).
Their efforts are individually and gratefully acknowledged [here](http://authors.galaxyzoo.org/). Thank you.

MW, IVS, MB and AMS gratefully acknowledge support
from the UK Alan Turing Institute under grant reference
EP/V030302/1. IVS gratefully acknowledges support from
the Frankopan Foundation.

# References
<!-- References: Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? -->
