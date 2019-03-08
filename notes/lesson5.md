## Key Points
---

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

## TODO
---

- Interpretation of Collab Learner
* Collab_filter excel sheet
* Broadcasting in numpy and pytorch
* Understand python function with args, kwargs
* Nearest neighbour (clustering) of the entity embeddings of the movie and users for unsupervised learning of the dataset

* Learn to see gradients of all layers and visualize/analyze them to see whats going wrong (understand numerical optimization, gradient descent)
* Gradient Descent excel sheet and understand all SGD optimizers
* Do finite differencing with own data

* Collaborative filtering NN approach:
	* add genre, timestamp feature
	* different dropouts
	* more hidden layers
* Collaborative filtering with binary data using classification concepts
* Online collaborative filtering

- Visualize Matrix Multiplication better


## Reading & Exploring
---

* Books:
	* Neural Network & Deep Learning web book chapt.4

* Paper:
	* Weight Decay, ADAMw, Momentum
	* Kaiming He initialization [jefkine.com: Initialization of Deep Networks Case of Rectifiers]

* Blogs:
	* Yes you should understand backprop Medium
	* Structured Deep Learning [Medium Towards Data Science]
	* How do we train neural networks? [Medium] Good technical writing

* Pytorch functional & nn modules
* Deep learning on small dataset
* Notation as a tool for thought

* Jacobian and Hessian for optimizing with finite differencing
* Linear Interpolation: alpha(...) + (1-alpha)(...)
* L2 regularization / Weight Decay


## Questions
---

- How does dot product of factors represent the actual matrix?

- For collab filtering, we take two high dimensional categorical variable and try to find a mapping?

- For data with only binary values, what to do and how to do it?

- Why unique user ids are generated? See excel sheet neural network sheet[40 min mark]

- Linear vs Non linear layers in neural network? Non linear activation functions? Function of a function of a function?

- Average of the squares of gradients? Why not the average of the square of difference of the gradient?