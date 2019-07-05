# ArticleAppraiser
This repo documents the code used to rank novel machine learning journal articles.

Notable packages used include `textacy`, `spacy`, and `scholarly`.


# Defining Novelty
For a paper to be novel and useful, I posit that it has to be (1) **new** and (2) **impactful**.

In academia, it is generally regarded that the number of times a paper is cited is a strong indicator of the novelty of a paper. However, there are two limitations to using this citation data to predict novelty:
- A paper only starts accummulating citations months after it has been published.
- Citation data of journal articles was not provided for this data challenge.

### Metrics
Given the data provided, two proxies can be used to measure novelty, as defined below:
1. **Topic Score**: A score that is based on the **chronological order** of a paper within a subject field (i.e. topic) and the **number of papers** in said field.
2. **Author Score**: A score that is based on the **h-index of authors** of the paper.

The final Novelty Score is calculated based on the two metrics above. More details can be found in the Jupyter Notebooks.


# Jupyter Notebooks
I have organized the work I did into five distinct notebooks. I recommend looking through them in the order below.

1. `EDA.ipynb` contains exploratory data analyses.
2. `Topic Modelling.ipynb` contains the pipeline to extracta and assign topics to each journal article.
3. `Scholars.ipynb` parses the scraped scholars information.
4. `Scores.ipynb` outlines the metrics used to score the documents.
5. `Model Training.ipynb` contains code for model development.
