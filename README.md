# Movie-Recommendation-System

# Overview
This repository contains a movie recommendation system built using the MovieLens 1M dataset. The system implements collaborative filtering, matrix factorization, and content-based approaches to provide personalized movie recommendations. A hybrid model combines these methods for enhanced accuracy, with user feedback loops for fine-tuning. The project includes an interactive web interface deployed via Streamlit and a dashboard built with Flask, which integrates movie posters using the IMDb API. This system is ideal for researchers, developers, and enthusiasts exploring recommendation systems in the entertainment domain.
Dataset

# Source: MovieLens 1M dataset, provided by the GroupLens research group at the University of Minnesota.
Description: Contains 1,000,209 ratings of ~3,900 movies by 6,040 users, collected between 2000 and 2003. Ratings are on a 1–5 scale (whole numbers).
Files:
ratings.dat: UserID, MovieID, Rating, Timestamp (UserID::MovieID::Rating::Timestamp).
movies.dat: MovieID, Title, Genres (MovieID::Title::Genres).
users.dat: UserID, Gender, Age, Occupation, Zip-code (UserID::Gender::Age::Occupation::Zip-code).


Size: ~24 MB (uncompressed).
Key Features:
User-movie rating interactions.
Movie metadata (titles, genres).
User demographics (age, gender, occupation).


# Tasks:
Collaborative filtering (user-based and item-based).
Matrix factorization (SVD, NMF).
Content-based recommendations using movie genres and metadata.
Hybrid model combining collaborative and content-based methods.
Deployment with Streamlit and Flask, including IMDb API for posters.



# Dataset Structure
The dataset is organized as follows:

ratings.dat: User-movie rating interactions.
movies.dat: Movie metadata (title, genres).
users.dat: User demographic information.
Note: Files use :: as a separator and require preprocessing (e.g., conversion to Pandas DataFrames).

# Downloading the Dataset
Download the MovieLens 1M dataset from:

MovieLens 1M Dataset

Or use this Python script to download and extract programmatically:
import urllib.request
import zipfile
import os

url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
dataset_dir = "./data/movielens-1m/"
os.makedirs(dataset_dir, exist_ok=True)
urllib.request.urlretrieve(url, os.path.join(dataset_dir, "ml-1m.zip"))
with zipfile.ZipFile(os.path.join(dataset_dir, "ml-1m.zip"), "r") as zip_ref:
    zip_ref.extractall(dataset_dir)

Important: Ensure ~100 MB of free storage. Verify file integrity to avoid data corruption.
# Project Timeline
The project was developed over four weeks, with the following tasks and milestones:
# Week 1: Data Collection and EDA

Tasks:
Collect MovieLens 1M dataset.
Clean and preprocess data (handle missing values, normalize ratings, parse genres).
Perform exploratory data analysis (EDA):
Analyze user-item interactions (rating distribution, matrix sparsity).
Visualize ratings, genres, and user demographics using Matplotlib/Seaborn.




Deliverables:
Cleaned dataset in a structured format (e.g., CSV, pickle).
EDA report with visualizations (rating histograms, genre distributions).



# Week 2: Collaborative Filtering and Matrix Factorization

Tasks:
Build collaborative filtering models (user-based and item-based).
Train matrix factorization models (SVD, NMF).
Evaluate using RMSE, MAE, and Precision@K (e.g., K=10).


Mid-Project Review (End of Week 2):
Collaborative filtering models implemented.
SVD and NMF models trained and evaluated.
Metrics (RMSE, MAE, Precision@K) documented.



# Week 3: Content-Based and Hybrid Systems

Tasks:
Develop content-based recommendation system using TF-IDF on movie genres/metadata.
Create a hybrid system combining collaborative and content-based methods.
Fine-tune recommendations with user feedback loops.


Deliverables:
Content-based system implemented.
Hybrid model developed and tested.
Feedback loop mechanism integrated.



# Week 4: Model Comparison and Deployment

Tasks:
Compare all models (collaborative filtering, SVD, NMF, content-based, hybrid) using RMSE, MAE, and Precision@K.
Deploy an interactive web interface using Streamlit.
Build a Flask dashboard integrating IMDb API for movie posters.
Prepare a final report with charts and use-case scenarios (e.g., recommendations for new vs. active users).


# Final Project Review (End of Week 4):
Best model saved.
Test set results compiled.
Streamlit interface and Flask dashboard deployed.
Final report and presentation completed.



Deployment

# Streamlit Web Interface:
Provides an interactive UI for users to input their ID or preferences and receive movie recommendations.
Displays movie titles, genres, and predicted ratings.


# Flask Dashboard:
Visualizes recommendations with movie posters fetched via the IMDb API.
Shows side-by-side comparisons of user-selected movies and recommendations.


# IMDb API Integration:
Fetches movie posters and metadata (e.g., release year, director) using the OMDb API (https://www.omdbapi.com/).
Requires an API key for access.



# Installation and Usage

Prerequisites:

Python 3.7+.
Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, surprise, streamlit, flask, requests.
IMDb API key (e.g., from OMDb).
Storage: ~100 MB for dataset and model files.


Setup:

Clone the repository:git clone https://github.com/username/movielens-recommender.git
cd movielens-recommender


Install dependencies:pip install -r requirements.txt


Download and extract the MovieLens 1M dataset (see above).
Configure the IMDb API key in config.py or environment variables.


# Preprocessing:

Load and clean data using Pandas (handle missing values, parse genres).
Create user-item rating matrix for collaborative filtering.
Generate TF-IDF vectors for content-based recommendations.


# EDA:

Run eda.py to analyze rating distributions, genre frequencies, and user demographics.
Visualizations saved in the outputs/ directory.


# Model Training:

Run train_collaborative.py for user-based and item-based filtering.
Run train_matrix_factorization.py for SVD and NMF models.
Run train_content_based.py for content-based recommendations.
Run train_hybrid.py for the hybrid model.
Evaluate models using evaluate.py (outputs RMSE, MAE, Precision@K).


# Deployment:

Streamlit:streamlit run app.py

Access at http://localhost:8501.
Flask Dashboard:flask run

Access at http://localhost:5000.
Ensure the IMDb API key is set for poster retrieval.


# Directory Structure:
movielens-recommender/
├── data/
│   └── movielens-1m/        # Dataset files
├── outputs/                  # EDA visualizations and model outputs
├── app.py                    # Streamlit app
├── dashboard.py              # Flask dashboard
├── train_collaborative.py    # Collaborative filtering scripts
├── train_matrix_factorization.py  # SVD/NMF scripts
├── train_content_based.py    # Content-based scripts
├── train_hybrid.py           # Hybrid model script
├── evaluate.py               # Evaluation script
├── config.py                 # Configuration (e.g., API keys)
├── requirements.txt          # Dependencies
└── README.md



# Example Usage

Streamlit:import streamlit as st
from surprise import SVD, Reader, Dataset
import pandas as pd

st.title("Movie Recommendation System")
user_id = st.number_input("Enter User ID", min_value=1, max_value=6040)
if st.button("Get Recommendations"):
    data = Dataset.load_from_df(pd.read_csv("data/ratings.csv")[["userId", "movieId", "rating"]], Reader())
    model = SVD()
    # Logic to predict and display top-10 movies
    st.write("Top 10 Recommended Movies:")


Flask Dashboard:from flask import Flask, render_template
import requests

app = Flask(__name__)
IMDb_API_KEY = "your_imdb_api_key"

@app.route('/')
def dashboard():
    # Fetch recommendations and posters
    posters = requests.get(f"https://www.omdbapi.com/?i=tt0111161&apikey={IMDb_API_KEY}").json()
    return render_template('dashboard.html', recommendations=[...], posters=[posters["Poster"]])



# Challenges

Sparsity: User-item matrix has ~4.2% density. Matrix factorization helps address this.
Cold Start: New users/movies lack data. Use content-based or popularity-based fallbacks.
Scalability: Optimize for large datasets using sparse matrices or batch processing.
API Limits: IMDb API may have rate limits. Cache posters locally to reduce calls.
Evaluation: Precision@K varies with K. Test multiple values (K=5, 10, 20).

# Citation
If you use the MovieLens 1M dataset, cite:
@article{harper2015movielens,
  title={The MovieLens Datasets: History and Context},
  author={Harper, F. Maxwell and Konstan, Joseph A.},
  journal={ACM Transactions on Interactive Intelligent Systems (TiiS)},
  volume={5},
  number={4},
  pages={1--19},
  year={2015},
  doi={10.1145/2827872}
}

# References

Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 1–19. doi:10.1145/2827872
MovieLens 1M Dataset
Surprise Library
Streamlit Documentation
Flask Documentation
OMDb API

# Contact
For issues or questions, open an issue on this GitHub repository or contact the developer via GitHub.
