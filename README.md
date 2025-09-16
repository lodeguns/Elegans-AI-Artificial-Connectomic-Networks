# Elegans-AI - Artificial Connectomic Networks

This is the repository of the journal paper: 
### Elegans-AI: how the connectome of a living organism could model artificial neural networks.
Francesco Bardozzo *1 , Andrea Terlizzi *1 , Claudio Simoncini *1, Pietro Lio' *2, Roberto Tagliaferri *1
<br>*1 Neurone Laboratory, University of Salerno, IT
<br>*2 Computer Laboratory, University of Cambridge, UK
 

### Abstract
This paper introduces Elegans-AI models, a class of neural networks that leverage the connectome topology of the Caenorhabditis elegans to design deep and reservoir architectures. Utilizing deep learning models inspired by the connectome, this paper leverages the evolutionary selection process to consolidate the functional arrangement of biological neurons within their networks. The initial goal involves the conversion of natural connectomes into artificial representations. The second objective centers on embedding the complex circuitry topology of artificial connectomes into both deep learning and deep reservoir networks, highlighting their neural-dynamic short-term and long-term memory and learning capabilities. Lastly, our third objective aims to establish structural explainability by examining the heterophilic/homophilic properties within the connectome and their impact on learning capabilities. In our study, the Elegans-AI models demonstrate superior performance compared to similar models that utilize either randomly rewired artificial connectomes or simulated bio-plausible ones. Notably, these Elegans-AI models achieve a top-1 accuracy of 99.99% on both Cifar10 and Cifar100, and 99.84% on MNIST Unsup. They do this with significantly fewer learning parameters, particularly when reservoir configurations of the connectome are used. Our findings indicate a clear connection between bio-plausible network patterns, the small-world characteristic, and learning outcomes, emphasizing the significant role of evolutionary optimization in shaping the topology of artificial neural networks for improved learning performance. Keywords: Artificial connectomes, C.elegans connectome, Connectomic architectures, Deep connectomic networks, Deep neural network transformers, Echo-state transformers, Multi-dyadic motifs


<p align="center">
  <img width="700" height="350" src="./imgs/artificialelegans.png?raw=true">
</p>

 
**Source Code Availability**
 
If you use this code you must cite our paper.

[Here the Tensorflow Drive with code, weights and logs](https://drive.google.com/drive/folders/1oT3xghtkeap9c4LG3PtuAshs243D5IxV?usp=sharing)

The drive includes a Python model (compatible with Python >3.x) developed using the Keras API on TensorFlow 2.6.1. Within the repository, you will also find comprehensive server-side logs that detail the training and validation processes on an epoch-by-epoch basis. A consolidated summary (logs.csv) tabulates key performance metrics, specifically loss and accuracy for both training and validation datasets. Please note that the scripts provided are optimized for Linux Ubuntu systems (version 18 or later), and compatibility issues have been identified on Windows platforms. Additionally, the repository features the ConnectomeReader.py script, designed to parse GraphML files that encode the structural data of the C. elegans connectome, as well as connectomes generated via various simulation techniques. 

 
The source code distributed in this repository is under Apache Licence 2.0.
 

**Additional Code:**
The Pytorch code will be distributed in this repository under Apache License 2.0.
Pytorch implementation with Elegans TN as a classifier is released in our [NeuroneLab](https://github.com/NeuRoNeLab/connectome-livecell-cls) repository.

 

**Connectome Data**
The connectome mapping of C.Elegans is provided by Neurodata.io
[C.Elegans Connectome Data](https://neurodata.io/project/connectomes/)



**How to cite this paper**

```
@article{BARDOZZO2024127598,
title = {Elegans-AI: How the connectome of a living organism could model artificial neural networks},
journal = {Neurocomputing},
volume = {584},
pages = {127598},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.127598},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224003692},
author = {Francesco Bardozzo and Andrea Terlizzi and Claudio Simoncini and Pietro Lió and Roberto Tagliaferri}
}

@inproceedings{bardozzo2024fp,
  title={FP-Elegans M1: Feature Pyramid Reservoir Connectome Transformers and Multi-backbone Feature Extractors for MEDMNIST2D-V2},
  author={Bardozzo, Francesco and Fiore, Pierpaolo and Li{\`o}, Pietro and Tagliaferri, Roberto},
  booktitle={International Meeting on Computational Intelligence Methods for Bioinformatics and Biostatistics},
  pages={111--120},
  year={2024},
  organization={Springer}
}

@article{fiore192advancing,
  title={Advancing label-free cell classification with connectome-inspired explainable models and a novel LIVECell-CLS dataset},
  author={Fiore, Pierpaolo and Terlizzi, Andrea and Bardozzo, Francesco and Li{\`o}, Pietro and Tagliaferri, Roberto},
  journal={Computers in biology and medicine},
  volume={192},
  number={Pt B},
  pages={110274}
}

```
