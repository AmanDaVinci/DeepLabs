## Key Points
---
* Neural Networks are better represented as matrix multiplications than neurons

* Activations are just the output number(single number) after input is multiplied with weights or filters or whatever   

* [EXP] Activation functions make the network non-linear. Matrix multiplications are only linear.

* [IMP] Even a shallow neural network can learn any function but the architecture must be so designed as to make it easy for it to learn. For example, softmax helps the network output a probability. It could learn that on its' own but defining it beforehand frees the network to do other critical learning.

* [TIP] Anthromorphize activation functions. For instance, softmax tries to pick one which makes it suitable for single label classification.

* [TIP] Use multi-label classification in learning Amazon product images for DICE.

* When using learning rate finder, use the point where the loss is clearly decreasing where slope is highest and the point is near the stable or flat region. Never pick the flat region as it means that the loss was remaining almost constant and hence the learning rate was too small or slow

* F-Beta can help define the trade-off between precision and recall. F-measure is the weighted harmonic mean between precision and recall. F1 means both are equivalent. F3 

* Fastai has resize function in its data loader to resize the actual dataset to have images smaller than a certain resolution. For example, standard images are 512x512 but certain images are 4096x4096, so we just resize them and bring them down to 1036x1036. Nothing to do with the size of the images during train time which is adjusted by sz.

* SGDR
	Cycles are how long we ride the wave from max to min; When cycle length is 1 Epoch we do the max to min in every one epoch. However, with cycle multiplication we increase the length of cycle to more than one epoch.
	Hence, with number of cycles = 3, cycle length = 1 epoch and multiplication = 2, we have:
	1st cycle with 1 epoch = 1
	2nd cycle with 1x2 epoch = 2
	3rd cycle with 1x2x2 epoch = 4
	Total Cycles = 3
	Total Epochs = 7

* "Bigger Batch Bigger Learning Rate": Higher batch size can do with a higher learning rate since we are more confident of the direction of descent. With lesser batch size it is more stochastic and hence requires a lesser learning rate to be careful. With smallest batch size of 1 we are basically taking more random directions every iterations. Also, the loss for large batch size is less noisy (surefooted) whereas losses of smaller batch size are very noisy. (https://miguel-data-sc.github.io/2017-11-05-first/)

* Gradient Descent:
	* Full Batch gradient descent (BGD) guaranteed to converge to global minimum for convex error surfaces & to local minimum for non-convex surfaces
	* BGD performs redundant computations for large datasets, recomputes gradients for similar examples before each parameter update
	* SGD bypasses redundancy by performing one update at a time. Much faster and can be used for online learning
	* SGD performs frequent updates with high variance causing objective(loss) function to fluctuate heavily
	* SGD's fluctuation enables it to jump to new & potentially better local minima than BGD
	* SGD's fluctuations complicates convergence since it will keep overshooting even if it is at the minimum. Decreasing the learning rate will show the same final convergence behaviour as BGD. 
	* Mini-batch SGD:
		* reduces variance of parameter updates
		* makes use of highly optimized matrix operations, makes deep learning feasible

* Challenges of Vanilla Gradient Descent (SGD):
	* Choosing lr is difficult, small lr slows convergence, large lr causes divergence or fluctuationsa about minimum
	* Can use schedules for lr but schedules are pre-defined and do not adapt to the dataset at hand
	* Same lr for all parameter updates, which needs to high for rare features in a sparse dataset
	* Gets stuck at saddle points

* Gradient Descent Optimization:	
	* Momentum:
		* Accelerates SGD in the dimension whose gradients do not change directions (smooth, not jagged)
		* Damps down oscillations in those dimensions whose gradients keep changing directions (rough, jaggy)
		* Faster convergence in ravine areas where the surface curves more steeply in one direction than other.
	* Nestrov Accelerated Gradient:
		* NAG gives the momentum a look ahead. Computes the gradient at the position the weight parameter will be in the next step approximately
		* This keeps it from overshooting the minima due to gathered momentum in a particularly smooth direction
		* Such momentum based approaches adapt the updates to the slope of the loss function
	* Adagrad:
		* Adapts lr to the parameters, low lr for parameters associated with frequent features and high lr for parameters associated with rare features
		* General learning rate eta is adapted for every parameter based on the past gradients computed for that parameter
		* lr =0.01, no need to manually tune the learning rate
		* Accumulation of squared gradients in the denominator shrinks the learning rate over time and the algorithm almost stops learning
		* Well suited for huge but sparse dataset
	* Adadelta:
		* Accumulate past gradients upto some window, to stop aggressive monotonic decreasing lr. Solves radically diminishing lr
		* No need of a default lr, it is set by the root mean square of the parameter (formula for more info)
	* RMSprop:
		* Similar to Adadelta, solves radically diminishing lr of Adagrad
		* Divides the learning rate by an exponentially decaying average of squared gradients
		* lr set to default of 0.001
	* ADAM:
		* Uses exponentially decaying average of past squared gradients to adapt lr for each parameters
		* Also uses exponentially decaying average of past gradients, similar to momentum
		* Compares favourably to others in practice

* Visualization of Gradient Descent Optimizers:
	![SGD optimization on loss surface contours](contours_evaluation_optimizers.gif)
	![SGD optimization on saddle point](saddle_point_evaluation_optimizers.gif)

* What Gradient Descent Optimization to use:
	* Always use adaptive learning rates for sparse data
	* RMSprop is an extension of Adagrad that handles diminishing learning rates
	* ADAM adds bias-correction and momentum to RMSprop
	* ADAM is the best overall, slightly outperforms RMSprop towards the end as gradients become sparser

* Other tricks of backprop:
	* Shuffling datasets
	* Curriculum Learning: Sort training examples in a meaningful way (e.g. easy to hard)
	* Batch Normalization: Network is normalized every mini-batch and backproped after this operation. Regularizer (reduces need of dropout)
	* Early stopping: stop when validation error does not improve
	* Gradient Noise: Add gaussian noise to each gradient update. escape local minima, robust to poor initialization

* Other than gradient descent: (https://news.ycombinator.com/item?id=11943685)

* Local minima with flat basins tend to generalize better than a sharp minima since it is brittle to slight changes in the weight vs loss surface

* Snapshot Ensembles:
	* Deep neural networks have many local minimas with similar error rates but different kind of mistakes 
	* Ensembling is training neural networks with different initialization causing convergence to different solutions
	* Averaging over predictions from these models leads to drastic reduction in error rates


## TODO
---

* Deep Learning (categorical embeddings) on structured data

* Single file prediction
* Model for multi-label classification using sigmoid activation vs softmax
* Various architecture for lesson 1
* Analyse model summaries of vgg and resnet
* Play with dataloader, datasets, generator, iterator concepts
* Dataset(single) vs Dataloader(batch)s
* Learn more python: class inheritance, enumerate, zip
* Experiment with high learning rates and see failure to convergence and even real divergence; already done using lr.find.plot_loss(), need to mark no improvement zone and real divergence

- Batch Normalization
- Plot Loss Change source code
- Use learn.sched.plot_loss()
- Minimum size, Freeze, Train with optimal lr, Unfreeze, Train with differential lr, Increase Size, Repeat
- Play with F-Beta score

## Reading & Exploring 
---

* Paper: Snapshot Ensembles
* Paper: SGDR

- Blogs
	- Blog: Apil Tamang, A world class classifier for cats and dogs (err..., anything)
	- Blog: Pavel Surmenok, Estimating an optimal learning rate for deep neural networks
	- Blog: Visualizing learning rate vs batch size, miguel-data-sc.github.io
	- Blog: Yes, Convolutional Neural Nets do care about Scale. Understanding Fast.ai "progressive resizing" technique.
	- Blog: Radek' Blog, A practitioner's guide to pytorch
	- Blog: Anand Saha, Decoding resnet architecture

- Paper: Cyclical Learning Rates for Training Neural Networks

- keras_lesson1 notebook
- lesson2_image-models notebook
- conv-example excel
- otavio visualization

## Questions
---

* Why does Validation data generator in keras has shuffle turned to false? 