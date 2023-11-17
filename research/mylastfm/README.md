# Analysis of my Personal Last.fm Profile.

## Citation

This work has been published in research journals and conferences.
You can cite the following publications:

> Ramirez Castillo, J., Flores, M.J. & Leray, P. Predicting spotify audio features from Last.fm tags.
>  Multimedia Tools and Applications (2023). https://doi.org/10.1007/s11042-023-17160-5.

> Ramirez Castillo, J., Flores, M.J. & Nicholson, A.E. User-centric Music Recommendations.
> Proceedings of the 16th Bayesian Modeling Applications Workshop (@UAI2022) (BMAW 2022).


## Data gathering from APIs

To download data from Last.fm and Spotify APIs, see https://github.com/jramcast/mer/tree/main/scripts.

Those scripts download the data into a MongoDB database.
You must start your own MongoDB instance.

After the raw data is available in the database, you must preprocess the data.
This step produces CSV files.

## Preprocessing Downloaded Data into CSVs

See `preprocess.py`.

Run:

```sh
$ python preprocess.py
```

## Training Lastfm-to-Spotify model

See `train.py`.


## Status:

- I have preprocessed the data and stored the CSV files ready for training in `data/jaime_lastfm`.
- Training XBG and Naive Bayes has been done.
- The results of XBG and Naive Bayes experiments are stored in the `research/mylastfm/results.log` file.
- I have trained these models with the `research/mylastfm/train.py` script.
- **Work in Progress** Experimenting with the transformers library to predict one of the Spotify values.
- A draft script is ready to train with transformers in `research/mylastfm/training/models/transformer.py` file.
- The draft script stores results in the `research/mylastfm/transformer_results/` directory.
- The transformers training phase has been run in vast.ai.
- Integrated the transformers model module into the `train.py` script and train with the full dataset.
- Integrated all results into a single file or directory.
- Use every track play as a data row.
- **TODO**: Aggregate the data by hour
- **TODO**: Experiment with tag selection (feature selection)
- **TODO**: Experiment with ranking metrics
- **TODO**: Create an evaluation metric (how close are the tags of recommended tracks vs input tags?)
