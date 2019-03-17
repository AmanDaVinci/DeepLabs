## Key Points
---

* SGD:
	* Batch SGD: Full Dataset
		* guaranteed convergence to global minimum in convex error surface
		* guaranteed convergence to local minimum in convex error surface
	* Stochastic SGD: One example
		* faster and frequent updates
		* can be used for online learning
		* high variance causes heavy fluctuation
		* fluctuation helps jump to better minima
		* jumping out of the best minima can be reduced by annealing
	* Mini-bacth SGD: n examples, best of both worlds
		* reduce variance resulting in stable convergence
		* efficient GPU optimizations on batch/matrix operations

* Challenges of Vanilla SGD:
	* Saddle points in high-dim non-convex error surface
	* Choosing an appropriate learning rate
	* If using learning rate schedules, it doesn't adapt to dataset 

* SGD Optimizations
	* Momentum:
```
			update = 0.9*update + 0.1*p_grad
			params = params - lr*update
```
		* SGD oscillates in ravines, steep slopes along one dimension
		* Slows down convergence
		* Add fraction (0.9) of the last update to current update
		* reduced oscillation by summing previous updates
		* opposite direction updates cancels each other
		* same direction updates add to each other

	* NAG (Nesterov Accelerated Gradient):
```
			??
			params = params - 0.9*prev_p_grad
			params.backward??
			update = 0.9*prev_p_grad + 
```
		* anticipate the next slope to slow down if required
		* approximate position of the parameters in the next step
		* compute gradient and add momentum accordingly

	* Adagrad(Adaptive Gradient):
```
		G += p.grad**2
		update = torch.sqrt(lr/(G+1e-8)) * p.grad
		params = params - update
```
		* adapts learning rate to parameters
		* smaller updates for params with frequent features
		* larger updates for params with rare features
		* used for sparse data training
		* no need to manually tune learning rate
		* ends up with infinitesimally small lr

	* Adadelta
```
		??
```
		* fixes adagrad's vanishing learning rate
		* doesn't sum up all past gradients
		* uses fixed size window of gradients
		* no need to set default learning rate

	* RMSprop
		* same as Adadelta

	* Adam (Adaptive Moment Estimation)
		* combines momentum with adaptive learning rates
		* exponential decaying avg of past gradients (momentum)
		* exponential decaying avg of past squared gradients (adaptive lr)
		* adam behaves like a heavy ball with friction that prefers flat minima

* Which optimizer to use:
	* Adam best overall choice
	* Adaptive learning rate methods do not need lr selection
	* For sparse data, adaptive learning rates
	* RMSprop, Adadelta and Adam perform similarly with Adam slightly better

* Additional SGD strategies:
	* Hogwild!
		* parallel sgd updates across workers
		* no locking of parameters
		* with sparse data only fraction of parameters are updated
	* Shuffling:
		* training data is shuffled after every epoch
	* Curriculum learning:
		* supplying training data in a meaningful way
		* sort examples by increasing difficulty
	* Batch Normalization:
		* parameters are initialized with zero mean & unit variance
		* eventually disappears and training slows down
		* batchnorm as a part of the architecture to re-establish normalization
		* changes are backpropagated
		* functions as a regularizer as well
	* Early stopping:
		* monitor stalling of validation loss
		* stop if not improving for some time
	* Gradient noise:
		* add gaussian noise to each gradient update
		* helps deep models escape minima
		* helps against poor initialization

* Matrix multiplication can be represented as dot product of every pair of rows (vectors) in the two matrices being multiplied

* Collaborative filtering solved using Gradient Descent and not with Linear Algebra factorization methods

* Entity embeddings is the crux of deep learning on structured data. Collaborative filtering is doing embedding from matrices and doing matrix factorization, Try out the various embedding dimensionality. Need to understand the how many factors are required to model the physical system.

* Matrix weights with kerr initialization with sd dependent on number of factors

* Pytorch train models with dataloaders with one mini-batch at a time. User doesn't need to loop thru each sample in the mini-batch and if he does so, gpu acceleration cannot be used

* When creating a network, make the output such that it is easy for it to optimize

* Linear algebra approach to collaborative filtering for matrix factorization fails when the matrix is sparse since empty values are taken as zero which means that a user who didn't watch a movie doesn't like the movie. But for gradient descent approaches the empty values are not included in the loss.

* Embedding is computational optimization over one hot encoding multiplied with a weight matrix

* Finite Differencing: Way of calculating derivatives. Derivative can be thought of change in one variable over a small change in the dependent variable: dE/dx = (E_x1-E_x2)/0.001 where x1 is slightly different than x2. A computer never does anything continuous but does it at discrete steps. Similarly, we humans can or need to think of differentials or integrals (differentials) with examples in real numbers. Kind of like tricks to visualize higher dimensions

* Concept of Neural Network:
	* just non-affice functions (relu) sandwiched between affine functions (matrix multiplication) 

* Backpropation is taking derivative of each layer wrt to previous layer and multiplying all of them together. Nothing but chain rule applied to layers. PyTorch has autograd (automatic differentiation) which has all the diferentiation rules. Neural networks have no neuron activations but matrices multiplications, ReLu is nothing but throwing away the negative values.

* Refactor formulae like code when reading papers, abstract away whatever you can

* Momentum means that if the error is decreasing towards a certain then keep going faster towards that. Hence jump over little bumps if the general slope is downwards. Linear interpolation between tbe last gradient and new the one. Gradient Descent with Momentum takes time but the results and predictions are better than ADAM.

* Adaptive Learning Rate (ADAM) keeps track of the average of the squared of the gradients to understand the surfece. If the gradient changes a lot the squared part will be huge and the learning rate will be reduced. Akin to walking on a bumpy ground. And hence for a smooth surface we can move faster. ADAM has bigger learning rate if the gradient has been constant for a while. However, if the gradient has been smooth for some time then the squared will be smaller?? 

* New ADAM: Exponetial moving weighted average of loss

* Regularization (Weight Decay):
		* Extra term to the cost function lambda * sum of weight squares
		* Regularization makes the network prefer learning small weights
		* Large weights will be allowed only if they considerably improve the loss
		* A compromise between finding small weights and minimizing cost function
		* Smaller weights means that the network won't change behaviour too much with few random input changes
		* L2 has weights squared and L1 has absolute weights

* L2 regularization / Weight decay: Do not change the weights a lot if it doesn't increase the loss upto to some significant level. Helps to avoid overfitting

* Dot vs Cross Product:
	* Dot product, the interactions between similar dimensions (x\*x + y\*y + z\*z)
	* Cross product, the interactions between different dimensions (x\*y, y\*z, z\*x)
	* The dot product (vec(a) · vec(b)) measures similarity because it only accumulates interactions in matching dimensions

* Three layer groups in fastai by default:
	* One for newly added head
	* The pre-trained layers split in two

* Affine functions are linear functions + constant:
	* In 1D: y = Ax + b
	* linear transformation followed by a translation
	* like matrix multiplication or convolution

* Embeddings in code:
	* looking up an array is identical to matrix multiplication by a one-hot encoded matrix
	* array lookup is computationally efficient

* Latent features emerge in the trained neural networks which capture semantics

* Use encoding latin-1 (older datasets) when utf-8 does not work 

* Use scaled sigmoid (min to max prediction range) to assist the network in choosing within a valid range. Make the max slightly higher since sigmoid asymtotes at max and will never be able to reach max on its own.

* Try and run experiments to be a good practioner

* Take weights or activations of any layer and do PCA to analyse and understand  

* Python args & kwargs:
	* Python function with \*args take variable sized non-keyword arguments as a list or collection
	* Python function with \*kwargs take variable sized keyword arguments as a dict

* Weight Decay or L2 regularization:
	* Complexity of a model is more than its number of parameters:
	* Functions need to more high dimensional to capture the real world
	* More parameters mean more interaction
	* Reducing parameters to solve overfitting is not a good idea
	* Instead we can sum the square of all parameters and minimize that as well
	* Since the square sum will be proportionately larger we need to down scale
	* This scaling parameter is called weight decay
	* Optimal value of weight decay is 0.1
	* Rarely if wd=0.1 causes underfitting, use wd=0.01
	* L2 Regularization form (loss function)
		* ```wd * (w**2).sum```
	* Weight Decay form (gradient of loss function)
		* ```2 * wd * w ``` 

## TODO
---

- Notebook: lesson5-mnist-sgd
- Use list comprehension after update loop returns loss.item() to plot loss

- Blog: An overview of gradient descent optimization algorithms
- Cross Entropy excelsheet
- Optimizer excelsheet

- Write your own nn.Linear
- See gradients
- Write your own optim.SGD with momentum

* Read Pytorch optim.Adam
* Write your own optim.Adam with annealing

* Visualize & Analyze gradients to see what's wrong:
	* Learn to use Tensorboard
	* (https://www.datacamp.com/community/tutorials/tensorboard-tutorial)

* Collaborative filtering NN approach:
	* add genre, timestamp feature
	* different dropouts
	* more hidden layers
* Collaborative filtering with binary data using classification concepts
* Online collaborative filtering

- Visualize Matrix Multiplication better
- Interpretation of Collab Learner
- Collab_filter excel sheet
- Broadcasting in numpy and pytorch
- Understand python function with args, kwargs
- Nearest neighbour (clustering) of the entity embeddings of the movie and users for unsupervised learning of the dataset


## Reading & Exploring
---

* Books:
	* Neural Network & Deep Learning web book chapt.2,3

* Blogs:
	* Yes you should understand backprop Medium
	* Structured Deep Learning [Medium Towards Data Science]
	* How do we train neural networks? [Medium] Good technical writing

* Concepts:
	* L2 regularization / Weight Decay
	* Experimenting with entity embeddings for categorical variable
	* Super Convergence
	* Finite differencing

* Paper:
	* Weight Decay, ADAMw, Momentum
	* Kaiming He initialization [jefkine.com: Initialization of Deep Networks Case of Rectifiers]

* Jacobian and Hessian for optimizing with finite differencing
* Linear Interpolation: alpha(...) + (1-alpha)(...)


## Questions
---

- How does dot product of factors represent the actual matrix?

- For collab filtering, we take two high dimensional categorical variable and try to find a mapping?

- For data with only binary values, what to do and how to do it?

- Why unique user ids are generated? See excel sheet neural network sheet[40 min mark]

- Linear vs Non linear layers in neural network? Non linear activation functions? Function of a function of a function?

- Average of the squares of gradients? Why not the average of the square of difference of the gradient?