# Recommender Metrics Repo

* `data`: stores data sets
* `experiments`:
  * `metrics`: Experimenting with ranking metrics.
  * `4qcnn`: Experiment to predict emotion quadrants from audio with the 4q dataset.
  * `mylastfm`: Experiment for a recommendation pipeline with my own Last.fm library.

## Experiments

Status:

- I have preprocessed the data and stored the CSV files ready for training in `data/jaime_lastfm`.
- Training XBG and Naive Bayes has been done.
- The results of XBG and Naive Bayes experiments are stored in the `experiments/mylastfm/results.log` file.
- I've trained these models with the `experiments/mylastfm/train.py` script.
- **Work in Progress** Experimenting with the transformers library to predict one of the Spotify values.
- A draft script is ready to train with transformers in `experiments/mylastfm/training/models/transformer.py` file.
- The draft script stores results in the `experiments/mylastfm/transformer_results/` directory.
- **TODO**: Integrate the transformers model module into the `train.py` script and train with the full dataset.
- **TODO**: Integrate all results into a single file or directory.
- **TODO**: Run the transformers training phase in vast.ai.


