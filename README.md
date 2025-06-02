# Movie-Recommender-System
A personalized movie recommender built with PySpark and collaborative filtering

Just answer a few questions — your favorite genres, a movie you like, or how picky you are — and it’ll serve up a tailored watchlist. Bonus: fuzzy-matching handles your typos like a pro.

Features
Personalized movie recommendations using ALS (collaborative filtering)
Filters by preferred genres and minimum rating
Fuzzy search for movie titles using natural typing (e.g., “Avngrs” → “Avengers”)
Model evaluation with RMSE
Clean, tabulated output for easy reading
Saves the trained model for reuse

Technologies Used
Tool/Library	            Purpose

PySpark (MLlib)	          Machine learning and data processing

ALS (Spark ML)	          Collaborative filtering model

PrettyTable	              Tabular formatting in terminal

FuzzyWuzzy	              Fuzzy string matching for movie names

Python-Levenshtein	      Speed-up for fuzzy matching
