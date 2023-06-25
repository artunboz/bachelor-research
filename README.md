# Clustering Faces of Comic Characters

This repository contains the code used for [X].

Below is the structure of the repository.

## src

Contains the classes used in the face clustering pipeline, which is visualized below.

![face_clustering_pipeline](figures/face_clustering_pipeline.png)

The pipeline is made up of 3 steps:

1. feature extraction
2. (optional) dimensionality reduction
3. clustering.

Each step has its own package, and each package contains abstract classes that are
responsible with common tasks. These abstract classes can be extended to other methods.

Global features refer to feature extraction methods that produce fixed-size vectors for
each image (assuming that images have the same dimensions). Examples include HOG, and
LBP features. Local features refer to methods that produce a variable number of vectors
per image. These vectors are transformed into a single fixed-size vector using a vector
quantization method such as fisher vectors. Examples include ORB and SIFT features.

## face_extraction

Contains the code for the face extraction pipeline, which is visualized below.

![face_extraction_pipeline](figures/face_extraction_pipeline.png)

The pipeline is made up of two steps

1. panel extraction and text cropping
2. face detection and extraction

You can use `panel_extraction_and_text_cropping/run.py` to run the panel extraction and
text cropping steps of the pipeline.

The second step relies on https://github.com/barisbatuhan/DASS_Det_Inference/tree/main,
so it must be installed as a separate repository. You add the modules found in
`face_detection_and_extraction` inside that repository, and use
`face_detection_and_extraction/run.py` to run the face detection and extraction steps of
the pipeline.

## experiment_scripts

Contains scripts for running the steps of the face clustering pipeline. It also contains
other scripts that I used to produce the results of the paper.

## mmselfsup_simclr

For the training of SimCLR, I used mmselfsup library. The exact configuration I used is
contained in `training_config.py`. You can use `get_latent_vectors.py` to embed images
using the trained backbone. Refer to https://mmselfsup.readthedocs.io/en/latest/ for
more details on mmselfsup.