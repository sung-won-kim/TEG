# Task-Equivariant Graph Few-shot Learning (TEG)

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://kdd.org/kdd2023/" alt="Conference">
        <img src="https://img.shields.io/badge/KDD'23-brightgreen" /></a>
    <img src="https://img.shields.io/pypi/l/torch-rechub">
</p>

The official source code for Task-Equivariant Graph Few-shot Learning (TEG) at KDD 2023.

## Abstract 
Although Graph Neural Networks (GNNs) have been successful in node classification tasks, their performance heavily relies on the availability of a sufficient number of labeled nodes per class. In real-world situations, not all classes have many labeled nodes and there may be instances where the model needs to classify new classes, making manual labeling difficult. To solve this problem, it is important for GNNs to be able to classify nodes with a limited number of labeled nodes, known as few-shot node classification. Previous episodic meta-learning based methods have demonstrated success in few-shot node classification, but our findings suggest that optimal performance can only be achieved with a substantial amount of diverse training meta-tasks. To address this challenge of meta-learning based few-shot learning (FSL), we propose a new approach, the **T**ask-**E**quivariant **G**raph few-shot learning (TEG) framework. Our TEG framework enables the model to learn transferable task-adaptation strategies using a limited number of training meta-tasks, allowing it to acquire meta-knowledge for a wide range of meta-tasks. By incorporating equivariant neural networks, known for their strong generalization abilities in other domains like N-body modeling, TEG can utilize these strengths to learn highly adaptable task-specific strategies. As a result, TEG achieves state-of-the-art performance with limited training meta-tasks. Our experiments on various benchmark datasets demonstrate TEG's superiority in terms of accuracy and generalization ability, even when using minimal meta-training data, highlighting the effectiveness of our proposed approach in addressing the challenges of meta-learning based few-shot node classification.

## Requirements
- python=3.8.13
- pytorch=1.10.1
- numpy=1.23.4
- torch-geometric=2.0.3

## How to run
```
python main.py --dataset corafull --way 5 --shot 3
```
### Flags
`dataset` : Amazon_clothing, Amazon_electronics, corafull, dblp, coauthorCS  
`way` : N-way  
`shot` : K-shot  
`qry` : M-query  
