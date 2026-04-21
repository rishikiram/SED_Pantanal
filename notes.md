## Current State:
### Stats in ROC-AUC
Validation Score: **0.9613**
Training Soundscape Score (new data): **0.6278**
Kaggle submission Score: **0.689**

Trained for 10 batches.

Current thoughts

Model is performing well on validation data, but these clips are fromthe same audio files as the training data; just different frames. The model is not abstracting to the soundscape data. Some things to note:
- only ~75% of the training data files are being used as some of the audio clips are less that 5s long. Also the endings of files are not used as they dont end on an interval of 5s.
- Soundscape data is possibly more mixed than data training data. Some exploratory data analysis can confirm this. Also soundscape noise is probably different. 'fine tuning' the model on the labeled soundscape data might be a good strategy.
- clever data transformations that allow me to reuse training data could be a game changer
- psuedo labeling would give me access to a lot more soundscape data, almost certainly a game changer. I need to learn how this would work.
- I am not sure if making the model bigger is going to give better results, but it is definitely worth trying
- efficientNet pools over both dimentions. A true SED model would not do this, or at least do it more intentionally. Could be worth making a custom CNN to pool mainly in the frequency (not time) dimention.