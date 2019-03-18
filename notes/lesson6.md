## Key Points
---

* Batch Normalization:
	* normalizing (subtract mean, divibe by variance) input helps
	* similarly normalizing activations between layers must help as well
		* higher learning rates
		* higher stability of training (vanishing & exploding gradients)
		* less need of dropout or other regularization
	* layer -> batchnorm -> activation function
	* must normalize using mean & var calculated for every mini-batch
	* however only a simple normalization changes what a layer can represent
		* normalized input to sigmoid will only use the linear region (0 to 1)
		* thus it loses non-linearity
		* must include an option for identity transformation (out=in)
		* so that SGD can reverse normalization if it was optimal for loss
	* hence the batch norm layer is differentiable

* Use fastai datapart to generate timeseries features

* Recurrence (RNN) limitations:
	* real world data has day,week,month and other metadata
	* better to make features out of these
	* rnn do not do well on kaggle challenges

* In Pytorch a model and a layer can be called like a function. This means more modularity using standard OOPs concepts. Pytorch has discontinued variables and uses tensors instead for everything.

* Use numpy as default since it is widely supported, use pytorch only when gpu or differentiation is needed.

* Create tasks for unsupervised learning:
	* unsupervised learning is based on fake task with labels learning
	* For unlabelled datasets, invent some labels exploiting some knowledge of the data or the domain.
	* Like for NLP, make correct and incorrect sentences by replacing existing words with random words.
	* Word embeddings are based on linear models and are good for visualization and maths but poor in predictive power.

* Make data pipelines (functions) for train and test data preprocessing

* In a normal neural network diagram, connections are the weight matrix and neurons are activations or outputs from the matrices before.

* Character level model and word level model can be combined with a bi-pair encoding

* RNNs are nothing but for loops where hidden and input states are concatenated

* Pytorch RNN input is sequence length(time steps) x batch size x input elements(hidden elements)

* To solve overfitting use regularization instead of reducing parameters

* Dropout:
	* deletes (zeros out) activations (not parameters) with some probability
	* use with caution

* Batchnorm:
	* does not reduce covariance shift aparently
	* instead makes the loss-vs-parameter landscape less bumpy
	* hence we can use higher learning rates with batchnorm
	* since it makes the loss surface smoother
	* In simple matrix terms:
```
			y_pred = f(params, x) * gamma + beta
			loss = y - y_pred
```
		where,
		gamma changes the scale of the activations
		beta shifts the activations around
		which,
		makes it explicitly easier for the SGD optimizer
		to scale & shift the output activations in the range of y

* Data Augmentation as regularization:
	* try all transformations on your dataset
	* pick those transformations which might be encountered realistically
	* out of padding modes reflection almost always works better 

* Research methodology:
	* When doing research, figure out things that you find interesting rather than following what others are working on
	* Ideas rarely come from math but mostly from intuition by brainstorming with physical analogies
	* be a practioner and experimentalist with ideas & intuitions
	* when it works, throw on some math to explain it

* high dimensions can be imagined by stacking 3d objects on top

* Convolution:
	* example:
		* input: height x width x channels
		* kernels: 3 x 3 x channels x 16 (kernels)
		* output: 16channels x height x width
		* stride 2 convolution
		* input: height/2 x width/2 x channels
		* kernels: 3 x 3 x channels x 32 (kernels)
		* output: 32channels x height/2 x width/2
	* we decrease dimensions using stride 2 convolution kerners
	* to increade channels using more such kernels
	* each channel represents one feature
	* height x width represents the position of that feature 
	* average pooling at the end of convolutions
		* the final activations are averaged across channels
		* height x width x channels -> 1 x channels

* fastai vision transformation:
	* max_lighting is the range of change beyond 0.5 in brightness function
	
## ToDo
---

* fastai Hook class

* Plot mean & std of activations & parameters to analyze training


- Pytorch Hooks
- Pytorch practice:
	- Grab one data item, normalize and put on GPU
	- create minibatch: add new dim to data
		- xb = x[None]; works with Numpy
		- xb = x.unsqueeze(0); pytorch specific, faster

- Notebook: lesson6-pets-more
	- data augmentation
	- convolution class activation map
---

* Notebook: lesson6-rossmann
* fastai date_part feature engineering
* preprocessors in fastai
* log differences in RMSPE

* Write a dropout layer (nn.Dropout):
	* visualize during train time
	* test during inference time

* model experiments:
	* tabular learner arch
	* dropout
	* sgd variants
	* embedding dropout
	* batch norm
	* summarize in excel
		* accuracy
		* variance of training (bumpiness of training)
		* smoothened graphs


---
- fastai tabular:
	- Subclassing in Python
	- TabularList
	- tabular_learner

* Use entity embeddings of rossmann from a trained neural network and use it with XGBoost or other simpler and quicker models. Visualize embeddings.  
* Rossmann challenge kernels for data visualization

* Bi-pair encoding for combining character and word level models
* Build a CharnModel with n = 3,4,5,6,7,8,9,10 and compare results

- Notebook: Lesson6-RNN, implement RNN


## Reading & Exploring
---

* Blogs:
	* Sebastian Ruder: Optimization for Deep Learning Highlights in 2017

* Ideas:
	* Create dummy task for unsupervised learning
	* Data Scientists experiment with features and data ideas a lot
	* Data augmentation for NLP or other domains

* Papers:
	* Batch Normalization
	* How does batch normalization help optimization

## Questions
---

* What is the rationale behind adding the extra parameters (other than mean and variance) to batch normalization?
	* The fastai lesson 3 notes seem to suggest that simply normalizing the activation outputs will not prevent the SGD to undo it in the next backpropagation. The original paper explicitly states that simply normalizing each input to a layer can change what it can represent. For example, feeding normalized inputs to a sigmoid will make use of the linear 0 to 1 window in the sigmoid or in other words, constrain the output to just the linear regime of the sigmoid non-linearity. Hence, they make sure that this transformation layer (batchnormalization layer) can also represent the idenity transformation. Since these new scale and shift parameters are learned rather than inferred from each mini-batch they have the power to restore the representation power of the layer. If need be, SGD can indeed use the scale and shift to undo the normalization completely if that is the optimal choice in reducing the loss. I have problem reconciliating the view of the notes that SGD will always try to undo it in the next steps and cause weight imbalance again. I believe it is not the right way to think about why the learnable parameters were added. 
