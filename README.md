# Recommendation Systems from First Principles

A detailed, self-contained Jupyter notebook that builds a complete recommendation-system learning example from scratch using only NumPy, pandas, and matplotlib.

## Overview

This notebook is designed for learners who are new to recommendation systems. Every major calculation — similarity scores, predictions, gradient updates — is implemented directly rather than pulled from a library, so the underlying math stays visible at every step. It works on a synthetic, deliberately sparse dataset (**50 users × 100 items**), so no external dataset or internet connection is required to run it.

## Learning goals

By the end of the notebook, you should understand:

1. How a user–item ratings matrix is created and why it matters.
2. Why most recommendation matrices contain many missing values (sparsity).
3. How cosine similarity measures similarity between users or items.
4. How user–user collaborative filtering makes predictions.
5. How item–item collaborative filtering makes predictions.
6. How matrix factorization decomposes a large matrix into two smaller ones via SGD.
7. Why cold start happens and how hybrid fallbacks help.
8. How offline recommendation quality is evaluated (RMSE and beyond).

## Requirements

- Python 3
- `numpy`
- `pandas`
- `matplotlib`
- `IPython` (for `display`)

Install with:

```bash
pip install numpy pandas matplotlib ipython
```

No dataset download is needed — all data is synthetically generated in-notebook with a fixed random seed (`RANDOM_SEED = 42`) for reproducibility.

## How to run

Open the notebook with Jupyter or JupyterLab and run all cells top to bottom:

```bash
jupyter notebook Recommendation_Systems_from_First_Principles.ipynb
```

The notebook is sequential — later sections (e.g., matrix factorization, evaluation, cold start) depend on variables defined earlier (e.g., the train/test split, similarity matrices), so cells should be run in order.

## Structure

The notebook is organized into 10 parts, roughly 44 numbered sections in total:

| Part | Topic |
|---|---|
| Setup | Imports, reproducibility, synthetic users/items/categories, hidden preference generation, sparsity injection, long-form conversion, data profiling and visualization |
| A | Simple baselines — global mean, user mean, item mean, popularity-based recommendation |
| B | Train/test split (80/20 per user) and the RMSE evaluation metric |
| C | Cosine similarity implemented from first principles |
| D | User–user collaborative filtering — similarity, prediction, explanation, Top-N recommendations, evaluation |
| E | Item–item collaborative filtering — same pipeline as Part D, applied to items |
| F | Matrix factorization — latent factors, biases, SGD training loop, loss curve, reconstruction, factor interpretation |
| G | Side-by-side comparison of all models against a global-mean baseline (RMSE bar chart) |
| H | Cold-start handling — new-user fallback via category preference, content-based similarity for new items, hybrid scoring |
| I | Small market-basket-analysis appendix (support, confidence, lift) |
| J | Industry view — how production recommender systems combine these components (candidate generation, ranking, re-ranking) |

The notebook closes with a **final recap** summarizing what was built and the conceptual distinctions between the approaches, followed by a set of **suggested learner exercises** (e.g., varying latent factor count, neighborhood size, adding Precision@K/Recall@K, building a hybrid recommender).

## Models covered

- **Popularity baseline** — Bayesian-adjusted average rating.
- **User–user collaborative filtering** — mean-centered cosine similarity between users.
- **Item–item collaborative filtering** — mean-centered cosine similarity between items.
- **Matrix factorization (SGD)** — `r̂_ui = μ + b_u + b_i + p_u · q_i` with 8 latent factors, trained over 60 epochs.
- **Content-based / hybrid fallback** — one-hot category and price-band similarity, blended with collaborative scores for cold-start cases.

All models are compared offline using RMSE on a held-out 20% test split per user.

## Notes

- All data (users, items, ratings, categories) is synthetically generated — there is no real-world data or external API dependency.
- Model hyperparameters (e.g., `N_FACTORS = 8`, `LEARNING_RATE = 0.01`, `REGULARIZATION = 0.03`, `EPOCHS = 60`, neighborhood sizes) are defined as constants near their relevant sections, making them easy to change for experimentation, per the suggested exercises at the end of the notebook.
