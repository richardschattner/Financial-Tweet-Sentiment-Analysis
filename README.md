# Financial-Tweet-Sentiment-Analysis
*Note that this repo is still a work in progress and uncompleted*

Sentiment Analysis of Financial Data (including Tweets) is very common.
However, I did not find any Sentiment Analysis models which predicted a sentiment per financial entity, but only models outputting a single sentiment per input.
This is fine for analysing news articles, financial Filings and letters to shareholders, which are written about a single financial entity.  

However, when working with Tweets, very often-times multiple financial entities are mentioned, so assigning the entire tweet a single sentiment is not always appropriate.
For example a tweet such as "$INTC has tough days ahead, after being outperformed by $AMD this long." reflects a negative sentiment toward Intel and a positive sentiment to AMD.  

In this project I aim at creating a compact transformer model capable of predicting the sentiment of each mentioned financial entity.

## The Model

This section describes preprocessing, tokenization and the model architecture, as well as design choices.

## Preprocessing and Tokenization

Stock Tickers in the Tweets are identified using simple regex matching (this may be expanded to more sophisticated NER models at a later date) and masked.
This is handled by the `mask_tickers()` and `batched_masking()` functions within `util.py`.
The functions transform a tweet into a masked tweet and a masking dictionary 
```python
# Original tweet
"$INTC has tough days ahead, after being outperformed by $AMD this long."

# Masked version + mapping
"[E1] has tough days ahead, after being outperformed by [E2] this long.",
{"[E1]": "INTC", "[E2]": "AMD"}
```
The model uses the DistilBert Tokenizer, to which the special masking tokens `[E1]`, `[E2]`,...,`[E5]` have been added, enabling the model to predict the sentiment of up to 5 mentioned Tickers per Tweet.
After tokenizing the masked Tweet, the function get_entity_positions() in util.py returns the positions of each masking tokens in the tokenized Tweet: `List[List[int]]`, containing a list of indices of each occurence of the masked Ticker for each Ticker within the tweet.

*Note that I described the preprocessing of a single input for simplicity, everything is implemented for batches of inputs.*

The masking is done for the purpose of unbiasing the sentiment prediction with respect to the mentioned entity.
The predicted sentiment should depend only on the context of the tweet itself.  

I do not want the prediction to be influenced by factors, such as, that past tweets mentioning a given company have been overwhelmingly positive.
By masking, the model cannot rely on the learned bias of each Stock Ticker, especially given that the training data comprises past tweets.  

The past performance of the stock will be reflected in the sentiment of its past tweets within the dataset.  
Past financial performance should not be used as an indicator of future performance, similarly I do not want past sentiments about a stock to affect the models future(present) sentiments about it.


## Model Architecture

Inputs to the model are (batched) masked tweets. 
The model itself is a basic encoder only model using bidirectional transformers.  
After a forward pass through the model, the embedding of the `[CLS]` token is the logit for the whole-sentence-sentiment.
This is only used for the first stage of pre-training.  

The sentiment logits of each entity is pooled from the embeddings of each mask token for that entity `[Ei]`, i = 1,...,5.
Applying softmax then yiels a three dimensional vector of sentiment probabilities where
- `idx 0`: Neutral  
- `idx 1`: Bullish  
- `idx 2`: Bearish

The basic model framework is contained within the file `model.py`.
Since I am working with Tweets, a very short context length for the models is sufficient.  
I am currently experimenting with different architectures and hyperparameters.


## Training

Given strong limitations for both clean training data and compute, I am forced to be restricted to simple models and low compute training techniques, which I wanted to experiment with anyways.
The model training consists of three stages.  

First, the model learns to predict the sentiment of an entire tweet using distillation learning. 
Open-Source models for this task are easily available, I used [FinTwitBERT-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment).  

Next the model is trained for the specific task of providing per entity sentiment predictions, first using simple Tweets built from the first datasets.
Finally, the model is fine-tuned using LoRA on a small sample of real tweets, mentioning multiple stocks, which have been manually labelled by myself.  

The datasets are contained within `utils.py` and the training functionality within `train.py`.  
*Note that I am still in the process of experimenting with the model architecture and hyperparameters of the first training stage. The second and third training stages are outlined below, but are still in active developement.*

## Distillation Training

The teacher model I use is [FinTwitBERT-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment)
Distillation Training is done on the same datasets the teacher was trained on: 
- [financial-tweets-sentiment](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment)
- [synthetic-financial-tweets-sentiment](https://huggingface.co/datasets/TimKoornstra/synthetic-financial-tweets-sentiment)

I created a wrapper class for this model `Teacher` within the `util.py` module, which handles loading and saving the model and tokenizer, as well as combines the tokenization and model call into a single call.  

My model is trained on the same objective as the teacher model, i.e. predicting the sentiment of an entire tweet.
Using distillation training, I hope to approximate the knowledge representations of the teacher model within my simpler model.
The model should learn features useful to its next two stages of training, when it is trained for per-Entity Sentiment prediction.

## Per-Entity Sentiment Pre-Training

Unfortunately, I did not find any financial tweet datasets, with per-entity sentiment labels.
This is unsurprising, as I also did not find any models trained for this task.
Therefore, I created a synthetic dataset, built from labelled tweets, by merging multiple tweets together.

```python
# Two labeled Tweets
("$AMD is looking strong!", "Bullish"), ("$NVDA is a great company, but its very overhyped", "Bearish")

# Joined Version
"$AMD is looking strong!. $NVDA is a great company, but its very overhyped", {"AMD": "Bullish", "NVDA" : "Bearish"}.
```
In the first training stage, where the model learned to mimic the embeddings of the `[CLS]` token of its teacher, in this stage, the model learns to adjust the embeddings of the masking tokens `[E1]`,`[E2]`,... to accurately predict their sentiments.

Unlike real tweets, the mentions for each stock ticker are still neatly contained within their respective original tweets and therefore there is no contextual relationship between the mentioned stock tickers in this training set.

## Fine-Tuning with LoRA

Finally, I create a small dataset of real tweets, mentioning multiple stock tickers and manually label them.  
The dataset then contains the contextual information missing from the dataset in the previous training stage, such as: stock a outperforms stock b, stock a and stock b are competing with no clear winner, etc.
As this dataset is small, I used LoRA to fine-tune the model data-efficiently for this task.

## Details on Model training, hyperparameter choices, evaluation metrics, etc.

*This section is a placeholder and will be filled in gradually.*
