## Keypoints
---

* Part1 explores best practices, regression and classification. Part2 will cover generative modelling and research topics

* Need to repackage or detach the hidden output of the RNN cells after one Backprop thru time (bptt) to avoid keeping the history of operations

* bptt is the sequence length for any sequential data; decided by looking at stability and performance

* Idea of using Mini neural nets (linear regression) inside neural networks

* GRU (Gated Recurrent Unit) has hidden forget gate and input remember gate. LSTM has an extra cell state along with the gates

* Using Fastai optimizers for pytorch models:
	LayerOptimizer has learning rate decay, weight decay and can also be used for callbacks such as SGDR (CosAnneal) or save in th end of every cycle

* Language models need patience and analysis, at some loss it is complete junk and at slightly better loss it suddenly looks right.

* Batch Normalization:
	Weight matrices can cause activations from matrix multiplication to explode. Normalizing the input with mean and std is one way but we also need to control the layer weights. With batch normalization, it normalizes all activations and then scales(product) it some parameters and increases(adds) with some parameters. It works because:
	* Normalization prevents gradient explosion
	* Scaling with weight gives the network an opportunity to scale only certain weight parameters

* Careful and rigorous experiments are more important to deep learning research than theory or practices 

* Use more modern techniques like CNN with 5x5 kernel size, no maxpooling but adaptive maxpooling at the end, keep exploring more techniques like batch normalization. 

* Resnet: based on boosting
	* y = x+f(x)
	* f(x) = y-x [residual]
	* only being used in computer vision now
	* preact resnet

* Class Activation Maps: final class vector multiplied with the output of the last conv layer

* Forward Hook[model.register_forward_hook(self.hook_fn)]: Callback that is run after every layer


## TODO
--- 

* Notebook: Lesson7-cifar10 	

* Use fastai to train custom models 

* BPTT and repackage_var()

* Learn pytorch sequential layering and class layers

* Study PyTorch model code

* Pytorch callbacks

* Notebook: Lesson6-RNN (stateful part)

* Notebook: Lesson7-CAM

* Understand torchtext concepts

* Ablation studies

## Reading & Exploring
---

* PyTorch Code:
	* RNNCell
	* LSTM needs init_hidden to return tuple

* Fastai Code:
	* ConvLearner.from_model_data(custom pytorch model, data)
	* LayerOptimizer
	* fit
	* CosAnneal (SGDR)

* Blog:
	* Colah's github: Understanding LSTM Networks
	* WildML: Recurrent Neural Network Tutorial Part 4-Implementing GRU/LSTM with python

* Kerem Turgutlu github Exploring Optimizers.ipynb

## Questions
---