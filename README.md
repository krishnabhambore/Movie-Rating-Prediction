# Movie Rating Prediction with Python

🔗 **Live project page:** https://krishnabhambore.github.io/Movie-Rating-Prediction/

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krishnabhambore/Movie-Rating-Prediction/blob/main/Movie_Rating_Prediction.ipynb)

A regression model that predicts the rating of an Indian movie from its genre, director, actors, release year, duration and votes.

This is Task 2 of a data science internship. The goal is to analyse historical movie data and build a model that estimates the rating users gave a movie, while learning which factors influence ratings.

## Dataset

[IMDb Movies India](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies) on Kaggle.

- 15,509 movies in the file, of which **7,919 have a rating** and were used for modelling
- Columns: Name, Year, Duration, Genre, Rating, Votes, Director, Actor 1, Actor 2, Actor 3
- Ratings average 5.84 and range from 1.1 to 10

## Approach

1. **Cleaning:** dropped movies with no rating, converted `Year`, `Duration` and `Votes` from text (like "(2019)" and "109 min") into numbers, and filled missing names and genres with "Unknown".
2. **Exploratory analysis:** rating distribution, average rating by decade and genre, and votes vs rating.
3. **Feature engineering:**
   - Genres became multi-hot columns (the 20 most common).
   - Each director and actor was replaced by the average rating of their movies (target encoding). This was calculated out-of-fold on the training data only, so the test set never leaks into the features.
   - Added the log of votes, the release year, the duration and the average score of the three actors.
   - Missing durations were filled with the median.
4. **Modelling:** an 80/20 train/test split (6,335 / 1,584 movies), then a Random Forest and a Gradient Boosting model compared against a baseline that always predicts the average rating.
5. **Evaluation:** RMSE, MAE and R² on the held-out test set.

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (mean rating) | 1.365 | 1.107 | -0.002 |
| Random Forest | 1.048 | 0.793 | 0.409 |
| **Gradient Boosting** | **1.027** | **0.773** | **0.433** |
| Gradient Boosting (no votes) | 1.121 | 0.863 | 0.324 |

Gradient Boosting performed best: on average its predictions are about **0.77 rating points** from the real rating.

### What influences ratings

Most important features (Random Forest):

| Feature | Importance |
|---|---|
| Average rating of the cast | 0.222 |
| Number of votes (log) | 0.190 |
| Release year | 0.152 |
| Director's track record | 0.117 |
| Duration | 0.066 |
| Lead actor's track record | 0.056 |

Genre had the smallest effect. Family, Musical and Drama scored highest by genre, and Action lowest, but the gap between genres was small.

## Limitations

- **Votes are only known after release.** The model without votes (R² 0.32) is the realistic "pre-release" version.
- **Predictions cluster in the middle.** Very low and very high ratings are pulled toward the average.
- **Newcomers have little history**, so scores for new directors and actors are less reliable.
- **Plot, budget and marketing are not in the data**, and adding them would likely improve accuracy.

## Future improvements

- Add more features such as budget, release season and plot text.
- Tune hyperparameters with cross-validation.
- Try XGBoost or LightGBM.

## How to run

1. Open `Movie_Rating_Prediction.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Click **Runtime → Run all**. The notebook downloads the dataset automatically with `kagglehub`.

To run it locally instead:

```bash
pip install -r requirements.txt
jupyter notebook Movie_Rating_Prediction.ipynb
```

## Repository structure

```
Movie-Rating-Prediction/
├── Movie_Rating_Prediction.ipynb   # full analysis and model
├── index.html                      # project page (GitHub Pages)
├── requirements.txt
└── README.md
```

## Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, kagglehub.

## Author

[krishnabhambore](https://github.com/krishnabhambore)
