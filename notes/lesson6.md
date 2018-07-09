## Key Points
---
* In Pytorch a model and a layer can be called like a function. This means more modularity using standard OOPs concepts. Pytorch has discontinued variables and uses tensors instead for everything.

* Use numpy as default since it is widely supported, use pytorch only when gpu or differentiation is needed.

* For unlabelled datasets, invent some labels exploiting some knowledge of the data or the domain. Like for NLP, make correct and incorrect sentences by replacing existing words with random words. Today unsupervised learning is based on fake task with labels learning. Word embeddings are based on linear models(simple matrix multiplication, quicker) and are good for visualization and maths but poor in predictive power.

* Make data pipelines (functions) for train and test data preprocessing

* In a normal neural network diagram, connections are the weight matrix and neurons are activations or outputs from the matrices before.

* Character level model and word level model can be combined with a bi-pair encoding

* RNNs are nothing but for loops where hidden and input states are concatenated

## ToDo
---

* Notebook: Lesson6-SGD, compute gradients using differentiation rules and also by finite differencing

* Notebook: Lesson6-RNN, implement RNN

* Use entity embeddings of rossmann from a trained neural network and use it with XGBoost or other simpler and quicker models. Visualize embeddings.  

* Rossmann Challenge kernels for data visualization

* Pytorch nn.Modulelist

* Bi-pair encoding for combining character and word level models

* Build a CharnModel with n = 3,4,5,6,7,8,9,10 and compare results

## Reading & Exploring
---
* Fastai Code:
	* learner.py: model property
	* column_data.py: CollabFilterModel
	* structured.py: load_model(), loads into current storage; could use to(device)
	* structured.py: MixedInputModel

* Blogs:
	* Sebastian Ruder: Optimization for Deep Learning Highlights in 2017

* Ideas:
	* Create dummy task for unsupervised learning

## Questions
---


