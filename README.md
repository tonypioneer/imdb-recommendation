# IMDb Movie Recommendation System

This repository aims to provide a system for movie recommendation via collaborative filtering techniques and produce additional visualizations. The approach is based on techniques like Singular Value Decomposition (SVD) and Neural Collaborative Filtering (NCF). The performance of the proposed system is evaluated using the IMDb datasets. This project is part of the Project Research for the COMP/ENGN8535 Engineering Data Analytics course at the Australian National University.

## Project Idea

The primary goal is to implement a system for movie recommendations using collaborative filtering techniques. The approach leverages methods like SVD and NCF, along with non-negative matrix factorization (NMF). The system's performance is evaluated using IMDb datasets.

- For SVD-based recommendations, refer to "Netflix Prize and SVD" by Stephen Gower: [Link](http://buzzard.ups.edu/courses/2014spring/420projects/math420-UPS-spring-2014-gower-netflix-SVD.pdf).
- For more information on NMF, read:
  - "Algorithms for Non-negative Matrix Factorization" [Link](https://papers.nips.cc/paper_files/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html)
  - "Learning the parts of objects by non-negative matrix factorization" [Link](https://www.nature.com/articles/44565)

After completing the movie recommendation task, the next step is to develop a data visualization method to display a small subset of the 1700 movies on a 2D plane to visualize their similarity or dissimilarity. Techniques like PCA, MDS, or k-means may be used to project 100 randomly selected movies of different genres on a 2D plane.

## Project Task

The task is to implement a movie recommendation system and produce additional visualizations using IMDb datasets.

## Environmental Setup

The experimental code is written in Python and runs in the following environment:

- Python 3.12
- Scikit-Learn 1.4.0
- Surprise 0.1
- TensorFlow 2.16.1

The hardware configuration is as follows:

- CPU: Apple Silicon M1 Pro
- RAM: LPDDR5 16GB
- GPU: Apple Silicon M1 Pro

## The File Structure
```
.
├── LICENSE
├── README.md
├── constants.py
├── data
│   ├── input
│   │   ├── movies.csv
│   │   └── ratings.csv
│   └── output
│       ├── result.csv
│       ├── test.csv
│       └── train.csv
├── main.py
├── recommender
│   ├── contextual.py
│   ├── knn.py
│   ├── movies.py
│   ├── ncf.py
│   ├── optimiser.py
│   ├── svd.py
│   ├── test.py
│   └── users.py
└── utils
    ├── dataset.py
    └── load_data.py
```

## Experiment

The experimental setup involves several machine learning algorithms and techniques to build and evaluate a movie recommendation system.

- In `main.py`, the integration of K-Nearest Neighbors (KNN) and Singular Value Decomposition (SVD) is tested for hybrid recommendations. The user and movie embeddings are reduced to 2D and 3D spaces using PCA and visualized with scatter plots to illustrate clustering behavior. K-means clustering is applied to identify user and movie clusters.
- The `test.py` script focuses on evaluating the recommendation system by calculating Root Mean Square Error (RMSE) to assess prediction accuracy.
- The `ncf.py` file implements Neural Collaborative Filtering (NCF), which leverages neural networks to capture complex user-item interactions. This model is also evaluated for RMSE.
- `contextual.py` integrates contextual information into the recommendation process, refining the recommendation quality.
- The `knn.py` script handles the KNN algorithm for both user-based and item-based collaborative filtering.
- The `optimiser.py` script focuses on optimizing the recommendation models for better performance.
- The `svd.py` script implements the pure SVD algorithm for generating recommendations.
- The `users.py` and `movies.py` files define the classes for user and movie KNN recommenders, respectively, and are responsible for computing similarity metrics and generating recommendations.
