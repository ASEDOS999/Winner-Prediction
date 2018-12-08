# Winner-Prediction
Project for winner prediction in the game "Dota 2" using the first few minutes

It uses machine learning for solving task. Using model is Logistic Regression.

##Data
Initial data was made using [YASP 3.5 Million Data Dump](http://academictorrents.com/details/5c5deeb6cfe1c944044367d2e7465fd8bd2f4acf). After preprocessing one gave file matches.jsonlines.bz2. There is field's description in folder 'data/dictionaries'.

##Features
Using features was exctracted by 'Features/extract_features.py'.

##Training
There are script ('dota_2_lr.py') and data ('features.csv') for training in folder 'training'.
