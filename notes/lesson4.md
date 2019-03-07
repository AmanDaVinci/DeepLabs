## Key Points
---

* Activation is just a number we get by multiplying inputs with weight

* Dropout is random throwing away a portion of the activations from the previous layers. This forces certain weights not to only recognize or overfit a part of the image. Solves the problem of overfitting and generalization for neural networks.

* Dropout is not applied to validation accuracy calculation. This makes the model give great accuracy at the start of the training.

* Implementation wise when doing dropout the remaining activations are increased in magnitude to compensate.	

* More parameters means more overfitting and hence we need to use higher dropouts. Start with 0.5 and increase or decrease depending on overfitting and 
underfitting.

* Taking log of softmax at the end layer works better for some reason

* Some continous data with less number of levels or variations can be made categorical. For example, day of the week. Categorical variables make the network think of them separately and not as a continous smooth function. The number of levels or classes of a categorical variable is called the cardinality.

* Mostly continous values are those with floating point numbers and having decimal points that would make the cardinality very high if they were categorical variables.

* For time series data, the validation set must be the most recent group and not random.

* Vector is Rank 1 Tensor

* Softmax is for categorical variables only. For regression values, throw away the softmax. Relu is never used at the last layer as it throws away the negatives.

* Forward Pass; Matrix multiplication changes the dimensionality of a vector.

```
INPUT VECTOR (batch x features)

(1 x 20)	->	[20 x 100] WEIGHT LAYER#1 (Parameters)
					|
					| ACTIVATION
					v
				(1 x 100)	->	[100 x 2] WEIGHT LAYER#2 (Parameters)
									|
									| ACTIVATION
									v
								(1 x 2) OUTPUT VECTOR
```

* The weights takes the rows-shaped input vector and spits out column-shaped output vector

* Embeddings for categorical variables in neural networks are better than hot encoding. They are trained along with the neural network. One hot decoding makes it linear with only one degree of freedom without making use of all the dimensions, whereas embeddings make full use of the high dimensions to represent the concepts.

* Categorical Embeddings heuristic rule is to divide categorical size by 2 but make sure not greater than 50: min(50, (sz // 2))

* Distributed Representation in high dimensionality is the most fundamental concept in neural networks. It does not have to mean one thing, it is a non-linear function to capture meaning in more than one dimension.

* [TIP] Categorical embeddings are better than continous values because the net can learn a high dimensional distributed representations instead of predicting a function for a single number.

* [TIP] Make as many features as possible in the beginning

* [EXP] Embeddings might work better than standard classical time-series

* Language is still behind vision with respect to deep learning approaches

* [EXP] Using transfer learning of a language model for Sentiment classsifcation. It might apply the learnt language model to help itself to output just a binary output of binary classification. First it learns to understand the structure of english and then it learns to recognize the sentiment in the language.

* Transfer learning or fine-tuning seems to perfrom better empirically. Need to see if there are studies on this.

* Label Encoding creates integer values for each class ('sun': 0, 'mon': 1, 'tue':2). Label Binarizer creates a one hot encoded vector for the multiclass labels ('sun': [1,0,0], 'mon': [0,1,0], 'tue': [0,0,1]) 

* Randomly choosing a validation set is not the best option always. Cross validation only works in those cases where you can randomly shuffle training data to get the validation set. Validation set should be such that it is not present in the training set and therefore easy. It must be more similar to real world test sets. Hence, validation set must be chosed rigourously.
	* You can check if your validation set is any good by seeing if your model has similar scores on it to compared with on the Kaggle test set
	* It can be instructive to see exactly what you’re getting wrong on the validation set, and Kaggle doesn’t tell you the right answers for the test set or even which data points you’re getting wrong, just your overall score.

* Neural Networks as Universal Function Approximator:
	* Matrix multiplication followed by matrix multiplication is just linear function on top of linear functions:
	y = ax + b & w = my + n -> w = (ma)x + (mb + n) [new slope & intercept]
	* However, adding an activation function which is non-linear makes it like stacking non-linearity over linearity which makes the entire operation non-linear

* Use transfer learning whenever possible: no point in starting with random weights

* For Kaggle NLP challenges use the test set during the language model training as well

* Activation Functions:
	* sigmoid: limits activation blowup but causes vanishing gradients
	* tanh: output is zero-centered so preferred over sigmoid, causes vanishing gradients & activation blowup
	* relu: avoids vanishing gradients, faster due to simpler function, causes dead neurons
	* leaky relu: solves dead neurons by having small negative slope
	* softmax: turns number into probabilities, paired with cross entropy loss

* Collaborative filtering has Cold Start problem
	* Solved by UX like Netflix
	* Solved by using a model trained on metadata

* For domain specific language modelling like Hindi in English or Pinyin for Mandarin, it must be trained on that language's corpus instead of wikitext103

* For tabular style timeseries RNN is not the best option, rather feature engineering in terms of day of week, seasonality, etc. are more practical

* Embedding matrix is a collection of feature vectors for each sample/user/movie/word

* To design a better architecture, make it easier to learn the right thing


## TODO
---

- lesson3-imdb
- lesson4-tabular
- collab_filter excel sheet
- Practice neural network terminology

- lesson4-collab
- Start exploring fastai with get_embed_learner

---

* Language generation systems using fastai - recent Jeremy tweet:
	* Transformer
	* Beam search
	* Pick trained model(domain) from Model-Zoo
* Find philosohpy on what can be learned by studying language alone?

* Check forums for solving Cold Start problem in fastai
* Random Forest to do hyperparameter searching in neural networks
* Use timestamp for collab filtering

---
* Experiment with various levels of dropouts in all the different layers. Find some rule of thumb. (1. Put same dropout on every single fc layer. 2. Put dropout only on the last layer)

* Categorical date embeddings to capture seasonality in timeseries

* Play with TorchText field and data loader. Observe how it processes each word or token

* Try transfer learning approaches on other NLP domains (text normalization)

---
- Data preparation with categorical, train test split, validation index split, null value filling, normalization, one hot encoding, read proc_df
- Experiment with and without dropouts. Observe generalization.
- Batch formation in language modelling (watch lesson7, early part)
- Using debugger in notebooks
- All activation functions
- Datablock api for Text
- Datablock api for Tabular
- Use transformer instead of LSTM (huge parameters)


## Reading & Exploring
---

* Blogs:
	* Self-supervised learning:
		https://towardsdatascience.com/a-different-kind-of-deep-learning-part-1-90fe6c52f1ab

- lesson3- Rossman
	* Very thoughtful data cleaning and feature engineering
	* Make summary for missing values and such
	* Join weather and state
	* Add datepart
	* Use germany data
	* Outer join all datasets
	* Fill missing values
	* Data cleaning for outliers and high cardinality
	* Durations until events like holidays
	* Set active index to date
	* Rolling quantiles(sum for 7 days)

- Blogs
	- How (and why) to create a good validation set, Rachel Thomas
	- Improving the way we work with learning rates, Vitaly
	- The Cyclical learning rate technique, Anand Saha
	- Differential Learning Rate, Manikanta



## Questions
---

* Why does average activation matter?
	The max pooling takes the maximum from a group of activations and if the dropout throws the maximum activation in that group, the output of max pooling changes which perpetuates to the next layer. Hence, we must try to keep the average activation same such as to keep the activations from getting too less in magnitude.

* Difference between categorical embeddings and one hot encoding?
	One hot encoding makes each category separate, linear and unrelated to the others. Embeddings can capture much richer representations and also relationships between the different categories. This gives the network a chance to learn the semantical meaning of a category in context of the data rather than a simple number. One hot encoding is also sparse and does not use the high dimensionality to go non-linear.

* Why does Dropout give better results in validation during the early training stages?
	It starts to overfit. We can experimentally find that with dropout the validation accuracy starts off bad but then the final accuracy after say n epochs is better than the final accuracy of a non-dropout model after n epochs, although it started with a better accuracy at first. If we look at the train and val loss we can find that this is due to overfitting.

* Why do neural networks perform better with scaled input?