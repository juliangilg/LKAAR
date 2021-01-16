# Localized kernel alignment-based annotator relevance analysis

This repository contains the implementation of our Localized kernel alignment-based annotator relevance analysis (LKAAR) model. The entire code is written in Matlab, which uses the library netlab (https://github.com/sods/netlab). The classification stage is based on Gaussian processes by using the GPML software (http://www.gaussianprocess.org/gpml/code/matlab/doc/).

LKAAR approach to estimate the performance of the sources in scenarios of supervised learning with multiple annotators. LKAAR computes the expertise of each annotator as the matching between the input features and the labels given by each source. Unlike previous approaches, LKAAR has three remarkable features: i) the performance of each annotator is a function of the input space; ii) the assumption of independence among the annotators is relaxed by modeling inter-annotators dependencies; and iii) the performance of the annotators is estimated using a non-parametric model, allowing it to be more flexible to the distribution of the labels. To the best of our knowledge, this is the first attempt to model both the dependencies among the annotators and the relationship between the input features and the labelers’ performance.

Pleae, if you use this code, cite this [paper](https://www.sciencedirect.com/science/article/pii/S0925231220316039?casa_token=Of51GiZn1LAAAAAA:xqgvDaBu7C9nDiEZ_DDs1aWvmlB_stG21NYSdQqI38aNBGcN_cagbr8h6hiFC5IPXpq9ftam)

@misc@article{gil2021learning,
  title={Learning from multiple inconsistent and dependent annotators to support classification tasks},
  author={Gil-Gonzalez, J and Orozco-Gutierrez, A and Alvarez-Meza, A},
  journal={Neurocomputing},
  volume={423},
  pages={236--247},
  year={2021},
  publisher={Elsevier}
}
