# Bayesian-Evaluation-of-Text-Classification-Models

When evaluating text classification models, we want to be certain about the performance of a model as well as its superiority over another. In the area of text classification it has become a norm to apply Null Hypothesis Significance Test(NHST) to statistically state and compare classifier performance. But, a frequentist approach has its own limitations and fallacies. In this report, we reflect on limitations posed by NHST. We also implement a novel Bayesian approach for evaluating text-classification models. We use a benchmark dataset and create several shallow models consisting of sparse and dense features and also an attention-based model for comparison.
We empirically demonstrate the difference between the two evaluation approaches.

## Project 
----

- [`Shallow_Models.ipynb`](https://github.com/S-B-Iqbal/Bayesian-Evaluation-of-Text-Classification-Models/blob/main/Shallow_Models.ipynb) implements feature engineering and modeling for NB, kNN,SVM,NNet, DT, RF and AB.
- [`FineTunedBert.ipynb`](https://github.com/S-B-Iqbal/Bayesian-Evaluation-of-Text-Classification-Models/blob/main/FineTunedBert.ipynb) implements DistilBERT over 5 epochs. Can take a few hours to train!
- [`PairedBootstrap.ipynb`](https://github.com/S-B-Iqbal/Bayesian-Evaluation-of-Text-Classification-Models/blob/main/PairedBootstrap.ipynb) implements frequentist NHST using paired-bootstrap.
- [`BayesianHypothesisTesting.ipynb`](https://github.com/S-B-Iqbal/Bayesian-Evaluation-of-Text-Classification-Models/blob/main/BayesianHypothesisTesting.ipynb) implements Bayesian Hypothesis test for comparing performance of model A and B.
- [`Figures.ipynb`](https://github.com/S-B-Iqbal/Bayesian-Evaluation-of-Text-Classification-Models/blob/main/Figures.ipynb) is used to generte several figures in the report.


## Notes
----
- PyTorch indexing was different from Sk-learn's indexing. In order to compare output of pytorch model with sklearn's output, we need to reset the index:

```python
# Example
sklearn.metrics.f1_score(ytest[ytest_bert_idx,:], ytest_pred_bert, average='micro', sample_weight=None, zero_division='warn')
```

- For NHST, the bootstrap sampling was not optimized, it can take a while to create 10000 bootstrap samples for each case!

## Datasets
----

All the model output are provided in [Data](https://github.com/S-B-Iqbal/Bayesian-Evaluation-of-Text-Classification-Models/tree/main/Data) folder.

To obtain feature-matrix from DitilBERT model, please refer the section "Creating BERT based features" in the [`Shallow_Models.ipynb`](https://github.com/S-B-Iqbal/Bayesian-Evaluation-of-Text-Classification-Models/blob/main/Shallow_Models.ipynb).

## Report
----
- PDF file in [Report](https://github.com/S-B-Iqbal/Bayesian-Evaluation-of-Text-Classification-Models/tree/main/Report) folder.
