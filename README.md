# MeLi Data Challenge 2021

*So, whether you eat or drink, or whatever you do, do all to the glory of God. (1 Corinthians 10:31)*

## The problem   
"Given the historical sales time-series for a subset of the martketplace's listings, we challenge you to predict how long will it take for a given item to run out of stock."     

So we had to output a probability distribution for each SKU, going from 1 to 30 days.   


## The solution    
I turned it into a classification problem, reshaping the data in such a way that
each SKU had only one row.    
I averaged the predictions from a LGBM model and a very simple Neural Network, that gave me 3.77443 in Private Leaderboard (6th place from 162 competitors).    
     

| Model     	    | Public LB 	| Private LB 	|
|-------------------|---------------|---------------|
| LGBM      	    | ~3.78     	|           	|
| Simple NN 	    | ~3.80     	|           	|
| Weighted Average	| ~3.76937  	| ~3.77443   	|



## Feature Engineering    
Instead of using `current_price`, I used it as a percentage change of price (I called it `pct_change`), because the currencies were different (Brazil, Mexico and Argentina).   
I created a binary feature called `has_zero_sold` as well.

Then I created some features using lag and rolling windows:

- Lag for `sold_quantity` and `minutes_active`

- Rolling mean for `sold_quantity`, `minutes_active` and `pct_change` 

- Rolling sum (count) for `has_zero_sold` and for the different categorical features (`listing_type`, `shipping_logistic_type`, `shipping_payment`)

For those rolling window features, I used window size of 19 days, keeping only the the 7 last windows: (30, 11), (29, 10), (28, 9), (27, 8), (26, 7), (24, 5), (23, 4).

At this time I was with ~100 features and was looking for some tips.        
I started to watch some tips from Giba in [this video](https://www.youtube.com/watch?v=RtqtM1UJfZc) and saw a method called LOFO (Leave One Feature Out).   
After applying LOFO, I reduced from ~100 features to 40 features and even got a small improvement on CV.    

For the LGBM model I used raw data and for NN I used standardized features, imputing NAs with zeros.

## Some insights
- Tuning LGBM model with Optuna decreased the difference between training and validation loss (in this case multilog loss), but the RPS stayed almost the same.   

- In the beginning I was creating week features and this was giving me ~4.03 in Public LB. After changing to rolling windows, the score improved to ~3.86. 

- When adding rolling counts for the categorical features (trying to capture changes of logistic status along the days for each SKU), scores improved from ~3.86 to ~3.80. 


## Reproducting    
To reproduce the solution, you must have the following files provided by MeLi:

- test_data.csv
- train_data.parquet
- items_static_metadata.jl

Now run the following steps:

1 - Run `1 - Preprocessing.ipynb` to preprocess the data and generate datasets.

2 - Run `2.1 - Optuna LGBM tuning.ipynb` to train a LGBM model and save out of fold predictions.

3 - Run `2.2 - Train LGBM.ipynb` to train the model using full data and generate a submission file.

4 - Run `3.1 - Train Neural Net, one layer (32).ipynb` to train a Neural Network and save out of fold predictions and the submission file.

5 - Run `3.2 - Train Neural Net, two layers (16, 16).ipynb` to train another different Neural Network and save out of fold predictions and the submission file.

6 - Run `4 - Average predictions.ipynb` to combine the predictions. This will generate the submission file called `4-average_predictions-2-tuned-lgbm-3.2-nn-two-layers-16-16-weights-052-048.csv`.

