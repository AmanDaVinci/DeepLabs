## Key Points
---

- Learing Rate is the key parameter in training neural networks. If learning rate is too high, the loss will shoot to infinity.

- The best way to improve a model is to use more data either by collecting more data or using data augmentation.

- We chose the learning rate where the decrease in loss is the highest, not where the loss is the lowest. This is because we cannot say anything about the lowest loss but if a learning rate causes the loss to decrease quickly that's what we were looking for in the first place. However, among all the learning rates where the slope(decrease in learning rate) are pretty high, we chose the one with highest magnitude due to the cyclical learning rate trick. The restart will guarantee that the lower learning rates will be tried along the way if we chose a learning rate which is highest amongst the good learning rates (with good slope in loss).   

- Activations: a single number, confidence or probability

- Precomputations: precomputed activations from the earlier frozen layers which are passed to the trainable (unfrozen) layer.

- Annealing Learing rate: step function (manual), sloped line (constant decrease) or half of cosine curve. Cosine Annealing: slow decrease at first(almost constant) then sudden drop when we are closer to the minima and then again slow decrease. The learning rate is changed after every single batch

- Stochastic Gradient Descent with Restarts (Cyclical Learning Rate): Earlier ensemble of models were trained in hope that one or more will find a good (very general or flat) minima. Now, we run restart the annealing process in hope that with a higher learning rate the model might jump out of the spikey or less general portion. Another idea is to save the weights (snapshot ensemble) at every lowest points in the learning rate cycle and later average the prediction from all saved weights.

- Precompute when transfer learning, unfreeze when learning from scratch, SGDR for better convergence and generalization, Differetial Learning Rate when kinda transfer learning

- SGDR with 3 cycles (1 cycle: highest to lowest lr), start with cycle length 1 epoch and then multiply cycle length with 2 epochs. If overfitting, use more cycle lengths with cycle mults for going over more epochs in one cycle

- Data augmentation in fastai is dynamic, it changes the image slightly at each epoch and hence each epoch sees a different version of the image

- Steps for world-class image classifier:
	- Enable Data augmentation and turn on precomputation
	- Find a decent learning rate using lr_find()
	- Train last layer with precomputed activations for 1-2 epochs	
	- Train last layer with data augmentation (precompute off) for 2-3 cycles with cycle length 1 epoch
	- Unfreeze all layers
	- Set earlier layers 3x to 10x (depends on the strength of transfer learning) lower than the next higher layer
	- Use lr_find() again to test if the last layers learning rate is still the best
	- Train full network with cycle multiplier of 2 epochs until overfitting

- If the scoring metrics are changing a lot at every run, the size of the dataset is the problem. For more granular control over accuracy where 3rd decimal of accuracy matters, we need more than a 1000 data points in our validation set since each point can change the metric we care about. A rule of thumb is that atleast 10-20 points must change class together to make a difference in the metric for it to be stable and reliable.

- Lower batch size makes the model more volatile since the gradient is computed on the smaller batch is less accurate. When GPU runs out of memory (CUDA out of memory error), we need to restart the kernel and reduce the batch size.

- The meaning and importance of accuracy differs with the number of classes. The random classifier baseline for 2 classes is 50% whereas the baseline for 10 classes is 10%.

- Start with small image sizes and then increase the image dimensions or size. This might give better results and prevent overfitting.

- Best way to handle unbalanced datasets in deep learning is to upsample (make more copies) the minor class.

* Some image features are scale invariant and some are not. Further, CNNs do improve with progressive resizing and training. Also with data augmentation. Why (scale invariance)?

* Regarding network size:
	The takeaway is that you should not be using smaller networks because you are afraid of overfitting. Instead, you should use as big of a neural network as your computational budget allows, and use other regularization techniques to control overfitting

* When thinking about production, we should do inference using a CPU first unless its a Google-scale problem. This will make it horizontally scalable, cheap and simple. With GPUs we can throw a bunch of input data as a batch to perform parallel inference. However, this requires us to setup a queue system for one batch to be generated as the users submit their input data and then feed it progressively.

* Training in real-time is not really required unless there exists some sort of concept drift.

* Learning Rate and Epochs:
	* If the validation loss shoots up really high, the learning rate needs to lowered
	* If the loss plots are going down really slowly, we must increase the learning rate
	* If the difference in training loss and validation loss is really huge, then increase the number of epochs or increase the learning rate
	* Train loss lower than validation loss is not a sign of overfitting, the validation error increasing again is a good sign of overfitting

* In python, \* is used to unpack a list (e.g. zip(\*ls) where ls is [[1,2,3],[a,b,c]]) and \*\* is used to unpack a dictionary (e.g. for passing args and values to a function using a dictionary)

* Three basic steps:
	* learn.fit_one_cycle(4, 3e-3)
	* learn.unfreeze()
	* learn.fit_one_cycle(4, slice(lr_find, 3e-4))

* Unbalanced data seems to work fine for Jeremy. But upsampling is suggested.

* A model is just a mathematical function, it doesn't take any space. The pre-trained weights are the coefficients or parameters of that function which are real numbers and do take space in memory.


## TODO
---

* Use fastai for production
* Try picking different lr from lr_find plot and see what works
* slice()

- Dictonary comprehension, zip(\* and \*\*) notation

- Async and await in python

- Use ImageDataBunch properly (get_tranfroms, valid_pct, size, normalize)

- Data cleaning after model training using its confidence and accuracy

- Improve on Cloud Classification by exploring fastai functions: data classes

- Use ?? to check documentation

- Try large cycle lengths of 4-16 epochs as in original CLR paper (might be long because they train from scratch, confirmed)

- Plot loss of SGDR to see the cycles finding better minimas

- Learning Rate Tricks: Annealing (decreasing as we get closer manually or using a function), Momentum, Cyclical Restarts-(http://ruder.io/optimizing-gradient-descent/)


## Reading & Exploring
---

* Notebook: lesson2-download
* Notebook: lesson2-sgd

- Paper: Cyclical Learning Rates for Training Neural Networks

- Notebook: lesson2-image_models


## Question
---
* If CNNs are transformation invariant how does data augmentation help at all? How does progressive resizing work? Aren't CNNs scale invariant? Then how does changing the scale help it learn better?

* If augmentation is not dynamic (not done randomly during run time), will precomputations work again? Also is fastai data augmentation dynamic and randomly generated for each train run?
	* ANS: Augmentation is dynamic. Hence, precomputation does not work with data augmentation. So we need to turn precomputation off or else it will use the cache with no augmentation

* Why do we crop the image to a standard size instead of padding it?
	* ANS: Experimentally did not work. CNNs do not find white borders interesting. Reflective padding works for satellite images. In practice, zooming seems to work better than padding. With padding the image can get arbitarily smaller. Fixed crop locations (sliding windows) might be better for test time augmentation.