# Winner-Prediction
Project for winner prediction in the game "Dota 2" using the first few minutes

It uses machine learning for solving this task. Using model is Logistic Regression.

## Data

An initial data was made using [YASP 3.5 Million Data Dump](http://academictorrents.com/details/5c5deeb6cfe1c944044367d2e7465fd8bd2f4acf). After preprocessing one gave file matches.jsonlines.bz2. There is a field's description in a folder 'data/dictionaries'.

## Features

Using features was exctracted by `Features/extract_features.py`.

## Training

There are a script (`dota_2_lr.py`) and a data (`features.csv`) for training in a folder `training`.
