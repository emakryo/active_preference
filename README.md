# Active Preference Learning Implementation

* based on [NIPS 2015](http://papers.nips.cc/paper/3219-active-preference-learning-with-discrete-choice-data) by Eric et al.

### modification

* without hyperparameter optimization by empirical Bayes
* variance prediction ($K^{-1}$ instead of $(K + C_{MAP}^{-1})^{-1}$)