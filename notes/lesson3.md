## Key Points
---
- Neural Networks are better represented as matrix multiplications than neurons
- Activations are just the output number(single number) after input is multiplied with weights or filters or whatever
- [EXP] Activation functions make the network linear. Matrix multiplications are only linear.
- [IMP] Even a shallow neural network can learn any function but the architecture must be so designed as to make it easy for it to learn. For example, softmax helps the network output a probability. It could learn that on its' own but defining it beforehand frees the network to do other critical learning.
- [TIP] Anthromorphize activation functions. For instance, softmax tries to pick one which makes it suitable for single label classification.
- [TIP] Use multi-label classification in learning Amazon product images for DICE.
- When using learning rate finder, use the point where the loss is clearly decreasing where slope is highest and the point is near the stable or flat region. Never pick the flat region as it means that the loss was remaining almost constant and hence the learning rate was too small or slow
- F-Beta can help define the trade-off between precision and recall. F-measure is the weighted harmonic mean between precision and recall. F1 means both are equivalent. F3 

**SGDR**
Cycles are how long we ride the wave from max to min; When cycle length is 1 Epoch we do the max to min in every one epoch. However, with cycle multiplication we increase the length of cycle to more than one epoch.
Hence, with number of cycles = 3, cycle length = 1 epoch and multiplication = 2, we have:
1st cycle with 1 epoch = 1
2nd cycle with 1x2 epoch = 2
3rd cycle with 1x2x2 epoch = 4
Total Cycles = 3
Total Epochs = 7


## TODO
---
- Analyse model summaries ($ learn) to see and understand architecture
- Train model on satellite/multi-label images using sigmoid activation at the last layer
- Various architecture for lesson 1
- Keras Lesson
- Play with dataloader, datasets, generator, iterator concepts
- Learn more python: class inheritance, enumerate, zip
- Plot Loss Change source code
-- Use learn.sched.plot_loss()
-- Minimum size, Freeze, Train with optimal lr, Unfreeze, Train with differential lr, Increase Size, Repeat
-- Play with F-Beta score

## Reading & Exploring 
---
- keras_lesson1 notebook
- Cyclical Learning Rates for Training Neural Networks
-- lesson2_image-models notebook
-- conv-example excel
-- otavio visualization

## Questions
---
- How does Data augmentation work if its' the same image? 
