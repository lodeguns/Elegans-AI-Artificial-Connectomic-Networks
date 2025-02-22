# Elegans-AI 
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
 
The source code distributed in this repository is under Apache Licence 2.0.
 

**Additional Code:**
The Pytorch code will be distributed in this repository under Apache License 2.0.
 

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
```
