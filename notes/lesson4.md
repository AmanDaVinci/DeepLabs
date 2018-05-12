## Key Points
---
- Activation is just a number we get by multiplying inputs with weight

- Dropout is randomy throwing away a portion of the activations from the previous layers. This forces certain weights not to recognize or overfit a part of the image.

- Dropout is not applied to validation accuracy calculation. This makes the model give great accuracy at the start of the training.

- Implementation wise when doing dropout the remaining activations are increased in magnitude to compensate.	

- More parameters means more overfitting and hence more dropouts

- Some continous data with less number of levels or variations can be made categorical. For example, day of the week.

- Categorical variables make the network think of them separately and not as a continous smooth function

- For time series data, the validation set must be the most recent group and not random.

- Vector is Rank 1 Tensor

- Softmax is for categorical variables only. For regression values, throw away the softmax

- Forward Pass; Matrix multiplication changes the dimensionality of a vector.

INPUT
(20 x 1)	->	[20 x 100] LAYER WEIGHT#1
					|
					| ACTIVATION
					v
				(100 x 1)	->	[100 x 2] LAYER WEIGHT#2
									|
									| ACTIVATION
									v
								(2 x 1) OUTPUT

- Categorical Embeddings for neural networks (better than hot encoding?)

- Categorical Embeddings heuristic is divide categorical size by 2 but make sure not greater than 50

- Distributed Representation in high dimensionality is the most fundamental concept in neural networks. It does not have to mean one thing, it is a non-linear function to capture meaning in more than one dimension.

- [TIP] Categorical embeddings are better than continous values

- [TIP] Make as many features as possible in the beginning

- [EXP] Embeddings might work better than standard classical time-series

- Language is still behind vision with respect to deep learning approaches

- [EXP] Using transfer learning of a language model for Sentiment classsifcation. It might apply the learnt language model to help itself to output just a binary output of binary classification. First it learns to understand the structure of english and then it learns to recognize the sentiment in the language.


## TODO
---
- Experiment with and without dropouts. Observe generalization.
- Experiment with various levels of dropouts in all the different layers. Find some rule of thumb.
- Data preparation with categorical, train test split, validation index split, null value filling, normalization, one hot encoding
- Categorical date embeddings to capture seasonality in timeseries
- Play with TorchText field and data loader. Observe how it processes each word/token
- Batch formation in language modelling

## Reading & Exploring
---
- articles by students
- entropy example sheet


## Questions
---
- Why does average activation matter?
- Difference between categorical embeddings and one hot encoding?
	One hot encoding makes each category separate, linear and unrelated to the others. Embeddings can capture much richer representations and also relationships between the different categories. This gives the network a chance to learn the semantical meaning of a category in context of the data rather than a simple number. One hot encoding is also sparse and does not use the high dimensionality to go non-linear.