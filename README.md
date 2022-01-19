# Bayesian-Evaluation-of-Text-Classification-Models


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
