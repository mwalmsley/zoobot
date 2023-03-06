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
    affiliation: 1
  # people who wrote core code and docs, alphabetical
  - name: Campbell Allen
    equal-contrib: true
    affiliation: 2
  - name: Ben Aussel
    corresponding: false
    affiliation: 3
    equal-contrib: true
  - name: Micah Bowles
    corresponding: false
    orcid: 0000-0001-5838-8405
    affiliation: 1
    equal-contrib: true
  - name: Kasia Gregorowicz
    orcid: 0009-0003-0023-6240
    corresponding: false
    affiliation: 4
    equal-contrib: true
  - name: Inigo Val Slijepcevic
    corresponding: false
    affiliation: 1
    equal-contrib: true
  # people who actively advised me (alphabetical)
  - name: Chris J. Lintott
    corresponding: false
    affiliation: 5
  - name: Anna M. M. Scaife
    corresponding: false
    affiliation: "1, 6"
  # heidelberg hackathon team + devina, who wrote experimental code/helped test TODO alphabetical
  - name: Maja Jabłońska
    corresponding: false
    affiliation: 4
  - name: Kosio Karchev
    corresponding: false
    affiliation: 7
  - name: Denise Lanzieri
    orcid: 0000-0003-2787-1634
    corresponding: false
    affiliation: 8
  - name: Devina Mohan
    corresponding: false
    affiliation: 1
  - name: David O’Ryan
    corresponding: false
    affiliation: 9
  - name: Bharath Saiguhan
    corresponding: false
    affiliation: 10
  - name: Crisel Suárez
    orcid: 0000-0001-5243-7659
    corresponding: false
    affiliation: ["11, 12"]
  - name: Nicolás Guerra Varas
    corresponding: false
    affiliation: 13
  - name: Renuka Velu
    corresponding: false
    affiliation: 14

affiliations:
 - name: Jodrell Bank Centre for Astrophysics, Department of Physics and Astronomy, University of Manchester, Manchester, UK
   index: 1
 - name: Zooniverse.org, University of Oxford, Oxford, UK
   index: 2
 - name: TODO
   index: 3
 - name: Astronomical Observatory of the University of Warsaw, Warsaw, Poland
   index: 4
 - name: Oxford Astrophysics, Department of Physics, University of Oxford, Oxford, UK
   index: 5
 - name: The Alan Turing Institute, London, UK
   index: 6
 - name: TODO
   index: 7
 - name: CEA, CNRS, Université Paris-Saclay, Université Paris Cité,  Paris, France
   index: 8
 - name: Vanderbilt University, Nashville, USA
   index: 11
 - name: Center for Astrophysics | Harvard & Smithsonian, Cambridge, USA
   index: 12
date: 03 March 2023
bibliography: paper.bib

---

# Summary

<!--  Summary: Has a clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided? -->

`Zoobot` is a Python package for measuring the detailed appearance of galaxies in telescope images
using deep learning.
`Zoobot` is aimed at astronomers who want to solve a galaxy image task such as finding merging galaxies or counting spiral arms.
Astronomers can use `Zoobot` to adapt (finetune) pretrained deep learning models to solve their task.
These finetuned models perform better and require far fewer new labels than training from scratch [@Walmsley2022Towards].

The models included with `Zoobot` are pretrained on up to 92 million responses from [Galaxy Zoo](www.galaxyzoo.org) volunteers.
Each volunteer answers a series of tasks describing the detailed appearance of each galaxy.
`Zoobot`'s models are trained to answer all of these diverse tasks simultaneously.
The models can then be adapted to new related tasks.

`Zoobot` provides a high-level API and guided workflow for carrying out the finetuning process.
The API abstracts away engineering details such as efficiently loading astronomical images, multi-GPU training, iteratively finetuning deeper model layers, and so forth.
Behind the scenes, these steps are implemented via either PyTorch or TensorFlow, according to the user's choice.
`Zoobot` is therefore accessible to astronomers with no previous experience in deep learning.

For advanced users, `Zoobot` also includes the code to replicate and extend our pretrained models.
This is used routinely at [Galaxy Zoo](www.galaxyzoo.org) to scale up galaxy measurement catalogs [@Walmsley2022decals]
and to prioritise the galaxies shown to volunteers for labelling.
Zoobot models have been applied to measure galaxy appearance in SDSS [@Walmsley2020], Hubble, HSC, and DESI, and are included in the data pipeline of upcoming space telescope Euclid [@2011arXiv1110.3193L].
We hope that `Zoobot` will help empower astronomers to apply deep learning to answer their own science questions.

# Statement of need
<!-- A statement of need: Does the paper have a section titled ‘Statement of need’ that clearly states what problems the software is designed to solve, who the target audience is, and its relation to other work? -->
<!-- State of the field: Do the authors describe how this software compares to other commonly-used packages? -->

One common way to investigate why galaxies look the way they do is by measuring
the appearance - morphology - of millions of galaxies and looking for connections between appearance and other physical properties [@Masters2019a].
The sheer number of images requires most of these measurement to be made automatically with software [@Walmsley2020].

Unfortunately, making automated measurements of complicated features like spiral arms is difficult because
it is hard to write down a set of steps that reliably identify those and only those features.
This mirrors many image classification problems back on Earth [@LeCun2015].
Astronomers often aim instead to learn the measurement steps directly from data
by providing deep learning models with large sets of galaxy images with labels (e.g. spiral or not) [@HuertasCompany2022].

Gathering large sets of labelled galaxy images is a major practical barrier.
Models trained on millions to billions of labelled images consistently perform better [@Bommasani2021;@https://doi.org/10.48550/arxiv.2302.05442], but astronomers cannot routinely label this many images.
Neither can most other people;
terrestrial practictioners often start with a model already trained ("pretrained")
on a broad generic task and then adapt it ("finetune") to their specific measurement task `[@https://doi.org/10.48550/arxiv.2104.10972].

Zoobot makes this approach available to astronomers.
We provide models pretrained on millions of galaxy images and present a convenient API for finetuning those models.
Early results [@Walmsley2022Towards;@https://doi.org/10.48550/arxiv.2303.00366] show that our pretrained models can be efficiently adapted to new morphology tasks.
<!-- We will continue to improve our pretraining methods to create adaptable galaxy "foundation" models and to make those models available to the community via Zoobot. -->

# Acknowledgements

Zoobot was made possible by the Galaxy Zoo volunteers,
who collectively created the crucial morphology labels used to create our pretrained models (and much, much more).
Their efforts are individually and gratefully acknowledged [here](http://authors.galaxyzoo.org/). Thank you.

MW, IVS, MB and AMS gratefully acknowledge support
from the UK Alan Turing Institute under grant reference
EP/V030302/1. IVS gratefully acknowledges support from
the Frankopan Foundation.

CS acknowledges the support NASA Heliosphysics Supporting Research Grant (NASA-HSR 16611153) and Vanderbilt Bridge Program PhD Fellowship.

Our (recommended) PyTorch [@Pytorch2019] version uses timm [@rw2019timm] for architecture definition, Pyro [@bingham2018pyro;@phan2019composable] for our custom loss and Pytorch Lightning [@Falcon_PyTorch_Lightning_2019] for training.
Our TensorFlow [@https://doi.org/10.48550/arxiv.1603.04467] version uses Tensorflow Probability [@https://doi.org/10.48550/arxiv.1711.10604] for our custom loss.

# References
<!-- References: Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? -->
