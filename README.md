# tensorflow-XNN

4th Place Solution for [Mercari Price Suggestion Challenge on Kaggle](https://www.kaggle.com/c/mercari-price-suggestion-challenge)

## About this project
This is the 4th text mining competition I have attend on Kaggle. The other three are:

* [CrowdFlower Search Results Relevance Competition](https://www.kaggle.com/c/crowdflower-search-relevance), 1st Place
* [Home Depot Product Search Relevance Competition](https://www.kaggle.com/c/home-depot-product-search-relevance), 3rd Place
* [The Hunt for Prohibited Content Competition](http://www.kaggle.com/c/avito-prohibited-content), 4th Place

In these previous competitions, I took the general ML based methods, i.e., data cleaning, feature engineering (see the solutions of [CrowdFlower](https://github.com/ChenglongChen/Kaggle_CrowdFlower) and [HomeDepot](https://github.com/ChenglongChen/Kaggle_HomeDepot) for how many features have been engineered), VW/XGBoost training, and massive ensembling. 

Since I have been working on CTR & KBQA based on deeplearning and embedding models for some time, I decided to give this competition a shot. With data of this competition, I have experimented with various ideas such as NN based FM and snapshot ensemble, which will be described in details below.

## Summary
This repo describes our method for the above competition. Highlights of our method are as follows:

* very minimum preprocessing with focus on end-to-end learning with multi-field inputs (e.g., text, categorical features, numerical features);
* hybrid NN consists of four major compoments, i.e., embed, encode, attend and predict. FastText and NN based FM are used as building block;
* purely bagging of NNs of the same architecture via snapshot ensemble;
* efficiency is achieved via various approaches, e.g., lazynadam optimization, fasttext encoding and average pooling, snapshot ensemble, etc.

Please find the slide of our solution [here](./doc/Mercari_Price_Suggesion_Competition_ChenglongChen_4th_Place.pdf).

## Preprocessing

## Multi-field Inputs

## Architecture
The design of the overall architecture of the model follows the well written post by Matthew Honnibal [1]. It mainly contains four parts as described below.

### Embed

### Encode
#### FastText
With FastText encoding [2], it just return the input unmodified.

#### TextCNN
TextCNN is based on [3]. It is time consuming based on our preliminary attempt. We did not go too far with it.

#### TextRNN
It is time consuming based on our preliminary attempt. We did not go too far with it.

#### TextBiRNN
Very time consuming.

#### TextRCNN
See Reference [4]. It is very time consuming.

### Attend
#### Average Pooling
Simple and effective. Works ok with FastText encoder.

#### Max Pooling
Simple but a litte worse than average pooling when works with FastText encoder.

#### Attention
We tried both context attention and self attention [5]. Context attention is very time consuming. Self attention helps to improve the results, but is also time consuming. At the end, we disable it.

### Predict
#### FM
We got many inspirations from the NN based FM models, e.g., NFM [6], AFM [7] and DeepFM [8]. Such FM models help to model the interactions between different fields, e.g., `item_description` and `brand_name`/`category_name`/`item_condition_id`.

#### ResNet
See Reference [9]. We started out with ResNet in the very beginning of the competition. While it works ok, it's a bit slower than pure MLP.

#### DenseNet
See Reference [10]. We also tried DenseNet besides ResNet. However, the decrease of RMSLE is ver mimimum. In the end, we stick with pure MLP, which is efficient and effecive.

### Loss
For loss, we tried mean squared loss and huber loss, and the later turned out to be a little bit better.

## Ensemble
In the final solution, we use cyclic lr schedule and snapshot ensemble to produce efficient and effective ensmble with a single NN architecture described above.
 
## Reference
[1] [Embed, encode, attend, predict: The new deep learning formula for state-of-the-art NLP models](https://explosion.ai/blog/deep-learning-formula-nlp)

## Note
Not finalized @ 2018.02.23.