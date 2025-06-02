from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import lit, col
from prettytable import PrettyTable
from fuzzywuzzy import process

def print_with_color(message, color_code):
    print(f"\033[{color_code}m{message}\033[0m")

def get_user_input():
    print_with_color("Welcome to the Movie Recommender System!", '1;32')

    while True:
        try:
            user_id = int(input("\nEnter your User ID (numeric value): "))
            if user_id <= 0:
                raise ValueError("User ID should be a positive integer.")
            break
        except ValueError as e:
            print(f"\033[91mInvalid input: {e}. Please try again.\033[0m")

    while True:
        try:
            num_recommendations = int(input("How many movie recommendations would you like? (e.g., 5, 10): "))
            if num_recommendations <= 0:
                raise ValueError("Number of recommendations should be a positive integer.")
            break
        except ValueError as e:
            print(f"\033[91mInvalid input: {e}. Please try again.\033[0m")

    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Horror", "Adventure"]
    print(f"\nAvailable genres: {', '.join(genres)}")
    genres_input = input("Enter your preferred genres (comma-separated): ")
    preferred_genres = [genre.strip() for genre in genres_input.split(",") if genre.strip() in genres]

    while True:
        try:
            min_rating = float(input("Enter your preferred minimum rating (between 1.0 and 5.0): "))
            if min_rating < 1.0 or min_rating > 5.0:
                raise ValueError("Rating must be between 1.0 and 5.0.")
            break
        except ValueError as e:
            print(f"\033[91mInvalid input: {e}. Please try again.\033[0m")

    movie_name = input("Enter a movie name you like (optional, leave blank to skip): ").strip()

    return user_id, num_recommendations, preferred_genres, min_rating, movie_name

def fuzzy_movie_search(movie_name, movies_df):
    movie_titles = [row['title'] for row in movies_df.select("title").collect()]
    best_match = process.extractOne(movie_name, movie_titles)

    if best_match:
        movie_title = best_match[0]
        print(f"\nBest match for '{movie_name}' is '{movie_title}'.")
        return movie_title
    else:
        print(f"\nNo close matches found for '{movie_name}'.")
        return None

def filter_by_genre_and_rating(movies_df, ratings_df, preferred_genres, min_rating):
    if preferred_genres:
        genre_filter = col("genres").rlike('|'.join(preferred_genres))
        movies_df = movies_df.filter(genre_filter)

    movies_df = movies_df.withColumnRenamed("movieId", "movie_id")
    filtered_ratings_df = ratings_df.join(movies_df, ratings_df["movieId"] == movies_df["movie_id"])

    if min_rating:
        filtered_ratings_df = filtered_ratings_df.filter(col("rating") >= min_rating)

    filtered_ratings_df = filtered_ratings_df.withColumn("userId", filtered_ratings_df["userId"].cast("integer"))
    return filtered_ratings_df, movies_df

def main():
    spark = SparkSession.builder.appName("Movie Recommender System").getOrCreate()

    print_with_color("\nLoading datasets...", '1;34')
    ratings_df = spark.read.csv("/content/rating.csv", header=True, inferSchema=True).dropna()
    movies_df = spark.read.csv("/content/movie.csv", header=True, inferSchema=True)

    print_with_color("\nPreview of available movie genres:", '1;34')
    preview_table = PrettyTable(["Title", "Genres"])
    for row in movies_df.select("title", "genres").take(5):
        preview_table.add_row([row["title"], row["genres"]])
    print(preview_table)

    user_id, num_recommendations, preferred_genres, min_rating, movie_name = get_user_input()
    filtered_ratings_df, movies_df = filter_by_genre_and_rating(movies_df, ratings_df, preferred_genres, min_rating)

    training_data, test_data = filtered_ratings_df.randomSplit([0.8, 0.2])

    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )

    print_with_color("\nPlease wait for a moment...", '1;33')
    als_model = als.fit(training_data)

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    predictions = als_model.transform(test_data)
    rmse = evaluator.evaluate(predictions)
    print(f"\nModel evaluation completed. RMSE = {rmse:.4f}\n")

    if movie_name:
        fuzzy_movie_search(movie_name, movies_df)
    else:
        user_movies = filtered_ratings_df.select("movieId").distinct().withColumn("userId", lit(user_id))
        user_recommendations = als_model.transform(user_movies)

        print_with_color(f"\nTop {num_recommendations} recommendations for User {user_id}:", '1;32')
        recommendations = user_recommendations.orderBy('prediction', ascending=False).limit(num_recommendations)
        recommendations = recommendations.join(movies_df, recommendations["movieId"] == movies_df["movie_id"])
        recommendation_table = PrettyTable(["Title", "Predicted Rating"])

        for row in recommendations.select("title", "prediction").collect():
            recommendation_table.add_row([row["title"], f"{row['prediction']:.2f}"])
        print(recommendation_table)

    als_model.write().overwrite().save("movie_recommender_model")
    spark.stop()
    print_with_color("\nThank you for using the Movie Recommender System!", '1;32')

if __name__ == "__main__":
    main()
