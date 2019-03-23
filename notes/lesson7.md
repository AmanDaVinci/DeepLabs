## Keypoints
---

* Transfer Learning is much better than Reinforcement Learning

* Part1 explores best practices, regression and classification. Part2 will cover generative modelling and research topics

* Recurrent:
	* Need to repackage or detach the hidden output of the RNN cells after one Backprop thru time (bptt) to avoid keeping the history of operations

	* bptt is the sequence length for any sequential data; decided by looking at stability and performance

	* GRU (Gated Recurrent Unit) has hidden forget gate and input remember gate. LSTM has an extra cell state along with the gates

	* Idea of using Mini neural nets (linear regression) inside neural networks

* Language models need patience and analysis, at some loss it is complete junk and at slightly better loss it suddenly looks right.

* Careful and rigorous experiments are more important to deep learning research than theory or practices 

* Use more modern techniques like CNN with 5x5 kernel size, no maxpooling but adaptive maxpooling at the end, keep exploring more techniques like batch normalization. 

* Batch Normalization:
	Weight matrices can cause activations from matrix multiplication to explode. Normalizing the input with mean and std is one way but we also need to control the layer weights. With batch normalization, it normalizes all activations and then scales(product) it some parameters and increases(adds) with some parameters. It works because:
	* Normalization prevents gradient explosion
	* Scaling with weight gives the network an opportunity to scale only certain weight parameters

* Resnet: based on boosting
	* y = x+f(x)
	* f(x) = y-x [residual]
	* only being used in computer vision now
	* preact resnet

* Class Activation Maps: final class vector multiplied with the output of the last conv layer

* Forward Hook[model.register_forward_hook(self.hook_fn)]: Callback that is run after every layer

* Using Fastai optimizers for pytorch models:
	LayerOptimizer has learning rate decay, weight decay and can also be used for callbacks such as SGDR (CosAnneal) or save in th end of every cycle

* Refactor code to minimize mistakes during research

* Densenets: input concat with next layer activations
	* works well with small datasets
	* memory intensive - huge activations but less parameters
	* good for segmentation since we want the original pixels to reconstruct

* Pretrained GAN:
	* pretrained generator & discriminator
	* training from scratch is like a blind leading a blind
	* pretraining solves the GAN training problem

* Where to use Unet:
	* generative modelling, semantic segmentation
	* input resolution is same as output resolution

* GAN is just a fancy loss function
	* can we come up with better loss functions
	* perceptual loss?

* RNN
```
INPUT WORD#1	EMBEDDING 
bs x 200 	->	[200 x 300]
					 |
					 v 			HIDDEN LAYER
				(bs  x 300) -> 	[300 x 100]
									 |
									 |
									 |
INPUT WORD#2	EMBEDDING   		 |
(bs x 200)	->	[200 x 300]			 |
					 v				 v 			HIDDEN LAYER
				(bs  x 300)	 +	(bs  x 300) ->	[300 x 100]
													 |
													 |
													 |
INPUT WORD#3	EMBEDDING							 |
(bs x 200)	->	[200 x 300]							 |
					 v								 v 			OUTPUT LAYER#3
				(bs  x 300)			+			(bs  x 300) -> 	[300 x 200]
																	 |
																	 |
																	 v
OUTPUT WORD#4					<-------						(bs  x 200)


```

* Extract from Dataloader
```
it = iter(data.valid_dl)
x1, y1 = next(it)
x2, y2 = next(it)
x3, y3 = next(it)
it.close()
```

* Validation set implementation:
	* does not need to be shuffled since the loss will be summed across bs
	* double the batch size: does not do backprop so grads are not stored

* PYTORCH:
	torch.nn
		* nn.Module:
		        * makes it into a callable as a function
		        * contains state like layer weights
		* nn.Parameters:
		        * tells the Module that these weights needs to be updated during backprop
		        * only Parameters with requires_grad=True are updated
		* nn.functional:
		        * contains loss functions, activation functions, etc
		        * contains non-stateful versions of convolution and linear layers

	torch.optim:
	    * coatains optimizers such as SGD
	    * step() to update weights with computed grads
	    * zero_grad() to zero out grads after each step 

	Dataset:
	    * wrapper around tensor objects with __len__ and __getitem__

	 DataLoader:
	    * Takes any Dataset and iterates over it in batches
	    * Implemented with a for loop over an iter object yielding in its body

## TODO
--- 

- Pytorch Tutorial by Jeremy

* Notebooks (top down approach)
	* lesson7-resnet-mnist
	* lesson7-superres-gan
	* lesson7-wgan
	* lesson7-superres
	* lesson7-human-numbers

* Visualize RNN batch and BPTT
* Visualize GRU

* Understand fastai model architecture
* Understand fastai model training callbacks

* Generate transforms using lists
* Explore data block at each step
* Replace two convs in a row with resnet block
* GAN hyperparameters

* Learn pytorch sequential layering and class layers
* Study PyTorch model code
* Pytorch callbacks


## Reading & Exploring
---

* Ideas:
	* think of different crappify functions

* Paper:
	* Deep Residual Learning, Kaiming He
	* Visualizing Loss Landscape of NN, Hao Li
	* Perceptual Losses for Real-Time Style Transfer and Super-Resolution

* PyTorch Code:
	* RNNCell
	* LSTM needs init_hidden to return tuple

* Blog:
	* Colah's github: Understanding LSTM Networks
	* WildML: Recurrent Neural Network Tutorial Part 4-Implementing GRU/LSTM with python

* Kerem Turgutlu github Exploring Optimizers.ipynb

## Questions
---

* Can we think of the .clamp_min(0.)-0.5 as essentially a version of Leaky Relu?
	* 