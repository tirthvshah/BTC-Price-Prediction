This code is part of an article that explores the potential use of the LSTM model for the purpose of Bitcoin price prediction.

Original article: [Using machine learning to predict future bitcoin prices](https://towardsdatascience.com/using-machine-learning-to-predict-future-bitcoin-prices-6637e7bfa58f?source=friends_link&sk=1406792b3ff9d2677c2ede4a1358f62e)

![*Photo by Chris Liverani on Unsplash*](https://miro.medium.com/max/1400/1*uVKMcAV-7uYXgIVUILbybA.png)

Is it possible to predict tomorrow’s Bitcoin price? Or if that’s too far a leap, what about the price 20 minutes from now?

I’ll be using the [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (LSTM) RNN machine learning model to predict the Bitcoin price 20 minutes from now, relying solely on simple historical financial data.

I’ve written this article partly as a guide, and partly as an excercise exploring the potential use of the LSTM model for the purpose of Bitcoin price prediction. Hence I may skip over some of the fundamentals, as these are easily found elsewhere.

A disclaimer that my experience comes primarily from curiosity, practical applications at a personal and professional capacity, and a [Fama’esque](https://sci-hub.tw/https://www.jstor.org/stable/2525569) intrigue into efficient markets. This may mean some of the terminology and methodology used could differ from others.

## A short primer on the LSTM model

What is the LSTM model exactly? In short it’s a form of recurrent neural network capable of learning long-term dependencies. In a similar fashion that we use prior experience to inform (preferably better) future outcomes, LSTM models use update gates and forget gates to randomly remember and forget pieces of historical information to inform their prediction.

The ability to use historical “context” like this allows these models to be particularly suited for prediction purposes, among a host of other applications.

To build a deeper understanding please take a look at [this](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) great resource which delves deeper into the exact machinations of this type of neural network.

## What tools we’ll be using

For this exercise I’m using [Numpy](https://numpy.org/install/) and [Pandas](https://pandas.pydata.org/getting_started.html) to deal with the data, and [Keras](https://keras.io/)/[Tensorflow](https://www.tensorflow.org/) for the machine learning functions. For debugging and its ability to present code nicely I use [Jupyter Notebooks](https://jupyter.org/install.html).

## Collecting the required data

To train our model we need training data. Any financial pricing data would suffice here as long as it’s available in minute intervals and is of a reasonable size. I’m using financial data spanning ~23 days with one-minute intervals.

I collected this data myself using the [Kraken API](https://www.kraken.com/features/api), so it’s likely it contains gaps. But it’ll do for illustration purposes. You can find my data, and the complete source code [here](https://github.com/derkzomer/predicting-future-bitcoin-prices).

{{< gist derkzomer 850b37ba5a76132b772eedfe6509b429 >}}

## Data preparation

First we start by importing all of the required packages, loading the dataset and removing the rows we are not interested in using.

{{< gist derkzomer e03725f83340924bfde16a6f2bf36072 >}}

We split the dataset up into a training and test set, and standardise its features. Standardisation is good practice as it reduces overfitting in cases where variance for some features may be higher than others.

{{< gist derkzomer e589fb629b77a710fb3325e9de997596 >}}

The LSTM model requires us to organise the data in blocks. Our data is grouped at one-minute intervals and we’ll use blocks of 50 minutes to predict the next block.

{{< gist derkzomer cf4910fe254f5a052f49dd445a03ff0e >}}

## The model

Now it’s time to train our model. We choose what type of model we want to use; sequential in this case, and we decide our hyper-parameters.

The model I’m using is relatively straightforward, containing 5 hidden layers with 50 neurons each, and a dropout in between every one of those hidden layers. We use the mean-squared-error loss function, the Adam optimiser, set the batch size at 32, and go through this network for 10 epochs.

Deciding on hyper-parameters is more art than science, and it’s worth testing out multiple options to understand what works best on your test data and in production. Optimising your hyper-parameters is outside of the scope of this article but some [great resources](https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/) exist online.

{{< gist derkzomer b2529236a91e1b918a91578af9daeef4 >}}

Now that we have a model that we can use to build predictions we can take a look at how it performs against our test data. The graph below plots the test data (y_test) prediction (y_hat) of both the ask price and bid price. As we can see the predictive values match the training data quite well.

![](https://cdn-images-1.medium.com/max/3372/1*8iznkIw3SxUJTWGW1VdiDw.png)

{{< gist derkzomer 03f807734a7a2274bb6aa072d4316d7b >}}

Cool! So this means I can predict Bitcoin prices now?

No. Not really. Unfortunately.

The predictive power of these models is definitely impressive, but if you take a closer look at the prediction you’ll see it quite narrowly follows the test data. The prediction appears to closely follow changes in price, but doesn’t often correctly forecast the price at the point at which these movements occur. This isn’t incredibly useful for any trading strategy you may want to use this for.

## Predicting the future

If we wanted to forecast the price 20 minutes from now one option would be to run the model over an immediately recent interval of historical data, concatenating the prediction to the end of our array of historical data, and then feeding that array back into the model — continuing this until we have 20 forecasted blocks of price predictions.

The below code does exactly that, and plots both the bid and ask price against the test data. Unsurprisingly it doesn’t really tell you much, with this particular prediction incorrectly forecasting a price increase.

![](https://cdn-images-1.medium.com/max/3368/1*1bm2MVKi5DxP2pTv4C1rIA.png)

{{< gist derkzomer 7e065991a3b714b06c020076dd80537c >}}

But will this be the case for the remainder of the test data? As the test data consists out of 6,661 rows we have 133 windows over which we can predict a trade.

Let us forecast the remaining 20 minute blocks, and sum the cumulative difference of the ask_price at n=4 and n=0.

The below plot shows you the cumulative gain of predicted *profitable* trades and actual trades, where one bought at the ask price of n=0 and sold at the ask price of n=4.

Interestingly it appears that the trades that provide you with an actual net benefit outweigh the ones that would have led to a loss. With a principal of $100 you’d come out ahead ~$31. Over just 4.5 days. That’s an astronomical return!

Likely too good to be true.

![](https://cdn-images-1.medium.com/max/3344/1*NsNoOT9MCux7l9HVjReC4A.png)

{{< gist derkzomer decf4c4b1aed545b807aa08780372060 >}}

Anyone that has ever bought BTC or any financial instrument on an exchange would now tell me you can’t actually sell at market at the ask price. The bid/ask spread would stop us from taking advantage of any increase in price, and only if the appreciation of the asset was higher than the spread would we be able to profit from this trade.

Once we model this into our prediction our golden goose starts to lose a few feathers. Our same $100 would still appreciate by ~$3.40! Still an impressive 1,318% annual return.

![](https://cdn-images-1.medium.com/max/3344/1*QBahzNuisRDPh4oU3iVj7A.png)

{{< gist derkzomer 435d0417cb8ff3c50323094c55e6e249 >}}

Now let’s add in transaction costs. We’ll take the liberal assumption that we can trade at market for the low fee of just 0.1% — reserved for traders trading at extremely large volumes.

It quickly becomes apparent our golden goose is not gold at all. Our model is only able to identify 3 potential profitable trades, of our same $100 would now turn a $2.26 loss.

![](https://cdn-images-1.medium.com/max/3352/1*RyT1DC-2cjekmZMgicZFXw.png)

{{< gist derkzomer f0a930bba76d3a198962b879f8f48b0c >}}

## Conclusion

The exercise above shows that simple financial data has some predictive power in forecasting short-term changes in price, but as there are no practical opportunities to profit from this information this particular model is relatively useless from a trading perspective.

Tools such as the LSTM model and others are becoming more accessible every day, with large groups and institutions pushing the boundaries of what these models can do through better data and superior processing capacity. This leads to markets integrating ever increasing amounts of information into asset prices — making arbitrage opportunities rare.

It isn’t a straightforward exercise to build a model with the predictive power to beat the market, and if one was able to easily use these tools to make a buck I certainly wouldn’t be sharing it with you here.
