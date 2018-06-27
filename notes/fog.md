# Fog Detection Notes
---

## Todo
---

* Request right side 5000 images
* Fog synthesis on 5000 images using matlab (octave did not work)
* Use Dilated CNN to train on clear view (started this)
* Transfer learning from clear view to fogged dataset

## Ideas
---

* Generate synthetic images using GANs (another semi-supervised approach)

* Dilated convolution is a simple but effective idea and you might consider it in these cases:
    * Detection of fine-details by processing inputs in higher resolutions.
    * Broader view of the input to capture more contextual information.
    * Faster run-time with less parameters

* RefineNet must be tried instead of DCN

* Better finetuning

* Better dehazing by predicting gamma of each image 

* Models perform better on test sets whose attenutation coefficient is similar to the one used for training which suggests that predicting this 'attenutation' parameter and then deploying the model for object detection and semantic scene understading is a good idea 

* Good semi-supervised approach, could be developed further

* Combine dehazing and semantic scene understading into a unified learned pipeline

## References
---

* http://www.erogol.com/dilated-convolution/ - for better semantic segmentation