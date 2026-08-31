# Amazon Last-Mile Route Clustering

## Project Description

This project is a command-line Python application that uses K-Means
clustering to group similar Amazon last-mile delivery routes.

Each record used by the clustering model represents one complete
delivery route.

## Dataset

The project uses the 2021 Amazon Last Mile Routing Research Challenge
Dataset from the Registry of Open Data on AWS.

Dataset website:

https://registry.opendata.aws/amazon-last-mile-challenges/

The dataset files are not included in this submission because they are
large. The included download_data.py program downloads the required
public training files into the data folder.

No AWS account, AWS access keys, or AWS CLI installation is required.

## Route Features

The clustering application uses the following route-level features:

- Number of delivery stops
- Number of packages
- Total planned service time
- Average stop latitude
- Average stop longitude
- Geographic spread
- Average package volume
- Vehicle capacity
- Route score

## Project Files

The main project files are:

- download_data.py: Downloads the required dataset files.
- find_dataset_files.py: Locates the correct files in the public S3 bucket.
- feature_engineering.py: Converts the original data into route-level features.
- main.py: Prepares the features, runs K-Means, and saves the results.
- requirements.txt: Lists the Python packages required by the application.

## Software Requirements

The project was developed using:

- Python 3
- PyCharm
- boto3
- pandas
- NumPy
- scikit-learn
- joblib

## Installation Instructions

Open the project in PyCharm.

Install the required Python packages by opening the PyCharm terminal and
running:

python -m pip install -r requirements.txt

If the python command is not recognised, try:

python3 -m pip install -r requirements.txt

## Dataset Download Instructions

Run download_data.py before running the clustering application.

In PyCharm:

1. Right-click download_data.py.
2. Select Run 'download_data'.
3. Wait for both files to download.
4. Confirm that route_data.json and package_data.json are in the data folder.

The downloaded dataset files are:

- data/route_data.json
- data/package_data.json

## Running a Small Test

Run main.py with the following script parameters:

--sample-size 50 --min-clusters 2 --max-clusters 5 --random-seed 42

The small test uses a sample of 50 delivery routes.

## Running the Main Experiment

Run main.py with the following script parameters:

--sample-size 500 --min-clusters 2 --max-clusters 8 --random-seed 42

The application tests different numbers of clusters and selects the
tested value with the highest silhouette score.

## Output Files

The application creates the following files:

- outputs/route_features.csv
- outputs/route_clusters.csv
- outputs/cluster_profiles.csv
- outputs/cluster_evaluation.csv
- models/kmeans_route_model.joblib

## Output Descriptions

### route_features.csv

Contains the engineered route-level features before clustering.

### route_clusters.csv

Contains every sampled route and the cluster assigned to it.

### cluster_profiles.csv

Contains the average characteristics of the routes in each cluster.

### cluster_evaluation.csv

Contains the silhouette scores produced for the tested numbers of
clusters.

### kmeans_route_model.joblib

Contains the fitted preprocessing steps and trained K-Means model.

## Main Application Entry Point

The main program is:

main.py

## Reproducibility

A random seed of 42 is used so that the same route sample and clustering
process can be reproduced.

## Dataset Attribution

2021 Amazon Last Mile Routing Research Challenge Dataset, accessed from
the Registry of Open Data on AWS:

https://registry.opendata.aws/amazon-last-mile-challenges/

The dataset is used for educational and non