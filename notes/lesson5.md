## Key Points
---

- Entity embeddings is the crux of deep learning on structured data

- Collaborative filtering is doing embedding from matrices and doing matrix factorization

- For each movie, we have some numbers that represent that movie. Same for users. When we do a dot product of the latent factors, we get the user to movie rating number.  
- Try out the various embedding dimensionality. Need to understand the how many factors are required to model the physical system.

- Conceptually, similarity between the embedding factors of all users and movies.

- Making collaborative filtering as a neural network with embedding dot product

- Matrix weights with kerr initialization with sd dependent on number of factors

- Pytorch do things one minibatch at a time and do not use loops during the forward function

- Adding bias to users and movies as a constant which decides if the movie is popular or the user is a movie buff.

- Array broadcasting, addition of vectors and matrices

- Sigmoid to transform an output from 0 to 1 and the multiply with maximum rating

- When creating a network, make the output such that it is easy for it to optimize

- Linear algebra approach to collaborative filtering for matrix factorization fails when the matrix is sparse since empty values are taken as zero which means that a user who didn't watch a movie doesn't like the movie. And the algebra approach takes on itself to put that as zero.

- Embedding is computational optimization over one hot encoding multiplied with a weight matrix

- Pytorch activation functions take an activation and return another activation. F has all the functions

- Observe the movielens notebook to see how to improve models and move from model to model.

- Finite Differencing: A computer never does anything continuous but does it at discrete steps. Similarly, we humans can or need to think of differentials or integrals (differentials) with examples in real numbers. Kind of like tricks to visualize higher dimensions

- Neural network is just a function of a function or a function

- Backpropation is nothing but chain rule applied to layers, neural networks have no neuron activations but matrices multiplications, ReLu is nothing but throwing away the negative values

- Momentum means that if the error is decreasing towards a certain then keep going faster towards that. Hence jump over little bumps if the general slope is downwards. Linear interpolation between tbe last gradient and new the one

- Gradient Descent with Momentum takes time but the results and predictions are better than ADAM

- Exponetial moving weighted average of loss

- Refactor formulae like code when reading papers, abstract away whatever you can

- Adaptive Learning Rate (ADAM) keeps track of the average of the squared of the gradients to understand the surfece. If the gradient changes a lot the squared part will be huge and the learning rate will be reduced. Akin to walking on a bumpy ground. However, if the gradient has been smooth for some time then the squared will be smaller??
And hence for a smooth surface we can move faster.

- ADAM is bigger learning rate if the gradient has been constant for a while.

- L2 regularization / Weight decay: Do not change the weights a lot if it doesn't increase the loss upto to some significant level. Helps to avoid overfitting



## TODO
---

- Start understanding fastai library

- Lesson 5: Movielens notebook

- Collaborative Filter excel sheet

- Gradient Descent excel sheet

- Visualize Matrix Multiplication better

- Nearest neighbour of the entity embeddings of the movie and users for unsupervised learning of the dataset

- Pytorch nn modules

- Dig into the various layers of abstraction of fastai

- Pytorch functional modules

- Add more hidden layers in the embedding network

- Play with neural network approach to collaborative filtering

- Do finite differencing with own data


## Reading & Exploring
---
 - Structured Deep Learning [Medium Towards Data Science]

 - Deep learning on small dataset

 - How do we train neural networks? [Medium] Good technical writing

 - Notation as a tool for thought

 - Gradient Descent excel sheet (read right to left)

 - Jacobian and Hessian for finite differencing

 - Linear Interpolation: alpha(...) + (1-alpha)(...)

 - L2 regularization / Weight Decay

k
## Questions
---

- How does dot product of factors represent the actual matrix?

- For collab filtering, we take two high dimensional categorical variable and try to find a mapping?

- For data with only binary values, what to do and how to do it?

- Why unique user ids are generated? See excel sheet neural network sheet[40 min mark]

- Linear vs Non linear layers in neural network? Non linear activation functions? Function of a function of a function?

- Average of the squares of gradients? Why not the average of the square of difference of the gradient?