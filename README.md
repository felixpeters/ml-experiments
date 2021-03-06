# ML Experiments

Central repository for playing around with new and exciting technologies

## Setup

Certain experiments require setting environment variables, e.g., for authentication
with services like Docker Hub, Paperspace, etc. See the `.env.example` file for
all environment variables that may be required. The techonologies/datasets below
list the required environment variables.

## Technologies

List of useful resources for finding new technologies:

- Awesome Production ML: [Github](https://github.com/EthicalML/awesome-production-machine-learning)

List of technologies used for the experiments:

- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Lifelines](https://lifelines.readthedocs.io/en/latest/)
- [pytest](https://docs.pytest.org/en/stable/index.html)
- [Hypothesis](https://hypothesis.readthedocs.io/en/latest/index.html)

## Datasets

List of data sources:

- IEEE-CIS Fraud Detection: [Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection/data)
- Insurance cost prediction: [Kaggle dataset](https://www.kaggle.com/mirichoi0218/insurance)
- Credit card customers: [Kaggle dataset](https://www.kaggle.com/arjunbhasin2013/ccdata)
- [News API](https://newsapi.org/): 
  - Open a free developer account to retrieve API key.
  - Add `NEWS_API_KEY=[YOUR_KEY]` to `.env` file.

## Planned experiments

Unsupervised learning:

- [ ] Apply [clustering techniques](https://scikit-learn.org/stable/modules/clustering.html) to [credit card dataset](https://www.kaggle.com/arjunbhasin2013/ccdata)
- [ ] Test [t-SNE algorithm](https://lvdmaaten.github.io/tsne/) (see [guide](https://distill.pub/2016/misread-tsne/) from Distill.pub)

## Issues

- [ ] Docker image: Enable GPU support for TensorFlow
- [ ] Documenatation: Add more detailed setup instructions
