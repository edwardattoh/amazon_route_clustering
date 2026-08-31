import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import build_route_feature_table, load_json


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
MODEL_DIRECTORY = PROJECT_DIRECTORY / "models"

ROUTE_DATA_FILE = DATA_DIRECTORY / "route_data.json"
PACKAGE_DATA_FILE = DATA_DIRECTORY / "package_data.json"

NUMERIC_FEATURES = [
    "number_of_stops",
    "number_of_packages",
    "total_planned_service_time_seconds",
    "average_stop_latitude",
    "average_stop_longitude",
    "geographic_spread_km",
    "average_package_volume_cm3",
    "vehicle_capacity_cm3",
]

CATEGORICAL_FEATURES = [
    "route_score",
]


def parse_arguments():
    """
    Read the settings supplied when the program is run.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cluster sampled Amazon delivery routes using K-Means."
        )
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of routes to sample. Default: 500.",
    )

    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Smallest number of clusters to test. Default: 2.",
    )

    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Largest number of clusters to test. Default: 8.",
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help=(
            "Use a fixed number of clusters. If this is omitted, "
            "the application chooses the best tested value."
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Controls reproducible sampling. Default: 42.",
    )

    return parser.parse_args()


def validate_arguments(arguments):
    """
    Check that the supplied settings are sensible.
    """
    if arguments.sample_size <= 0:
        raise ValueError(
            "--sample-size must be greater than zero."
        )

    if arguments.min_clusters < 2:
        raise ValueError(
            "--min-clusters must be at least 2."
        )

    if arguments.max_clusters < arguments.min_clusters:
        raise ValueError(
            "--max-clusters must not be smaller than "
            "--min-clusters."
        )

    if arguments.clusters is not None and arguments.clusters < 2:
        raise ValueError(
            "--clusters must be at least 2."
        )


def build_preprocessor():
    """
    Prepare the route features before K-Means is used.
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def test_cluster_counts(
    transformed_features,
    minimum_clusters,
    maximum_clusters,
    random_seed,
):
    """
    Test several values for the number of clusters.
    """
    number_of_routes = transformed_features.shape[0]

    effective_maximum = min(
        maximum_clusters,
        number_of_routes - 1,
    )

    if effective_maximum < minimum_clusters:
        raise ValueError(
            "The sample is too small for the requested "
            "range of clusters."
        )

    results = []

    for cluster_count in range(
        minimum_clusters,
        effective_maximum + 1,
    ):
        model = KMeans(
            n_clusters=cluster_count,
            random_state=random_seed,
            n_init=20,
        )

        labels = model.fit_predict(transformed_features)
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            score = np.nan
        else:
            score = silhouette_score(
                transformed_features,
                labels,
            )

        results.append(
            {
                "number_of_clusters": cluster_count,
                "silhouette_score": score,
            }
        )

    return pd.DataFrame(results)


def create_cluster_profiles(clustered_routes):
    """
    Create a summary describing every cluster.
    """
    numeric_profiles = (
        clustered_routes
        .groupby("cluster")[NUMERIC_FEATURES]
        .mean()
        .round(2)
    )

    cluster_sizes = (
        clustered_routes
        .groupby("cluster")
        .size()
        .rename("route_count")
    )

    route_score_modes = (
        clustered_routes
        .groupby("cluster")["route_score"]
        .agg(
            lambda values: (
                values.mode().iloc[0]
                if not values.mode().empty
                else "Missing"
            )
        )
        .rename("most_common_route_score")
    )

    profiles = pd.concat(
        [
            cluster_sizes,
            numeric_profiles,
            route_score_modes,
        ],
        axis=1,
    )

    return profiles.reset_index()


def run_application(arguments):
    """
    Run all stages of the route-clustering application.
    """
    validate_arguments(arguments)

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    MODEL_DIRECTORY.mkdir(exist_ok=True)

    print("Loading the downloaded JSON files...")
    print(f"Route data: {ROUTE_DATA_FILE}")
    print(f"Package data: {PACKAGE_DATA_FILE}")

    route_data = load_json(ROUTE_DATA_FILE)
    package_data = load_json(PACKAGE_DATA_FILE)

    print(
        f"\nSelecting a sample of "
        f"{arguments.sample_size} routes..."
    )

    feature_table = build_route_feature_table(
        route_data=route_data,
        package_data=package_data,
        sample_size=arguments.sample_size,
        random_seed=arguments.random_seed,
    )

    print(f"Routes in the sample: {len(feature_table)}")

    print("\nChecking for missing values:")
    print(feature_table.isna().sum().to_string())

    feature_table_path = (
        OUTPUT_DIRECTORY / "route_features.csv"
    )

    feature_table.to_csv(
        feature_table_path,
        index=False,
    )

    preprocessor = build_preprocessor()

    model_features = feature_table[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]

    print("\nPreparing the route features...")

    transformed_features = preprocessor.fit_transform(
        model_features
    )

    if arguments.clusters is None:
        print(
            f"\nTesting cluster counts from "
            f"{arguments.min_clusters} to "
            f"{arguments.max_clusters}..."
        )

        evaluation_results = test_cluster_counts(
            transformed_features=transformed_features,
            minimum_clusters=arguments.min_clusters,
            maximum_clusters=arguments.max_clusters,
            random_seed=arguments.random_seed,
        )

        valid_results = evaluation_results.dropna(
            subset=["silhouette_score"]
        )

        if valid_results.empty:
            raise ValueError(
                "A valid silhouette score could not be calculated."
            )

        best_row = valid_results.loc[
            valid_results["silhouette_score"].idxmax()
        ]

        selected_clusters = int(
            best_row["number_of_clusters"]
        )

        print("\nCluster-testing results:")
        print(evaluation_results.to_string(index=False))

        print(
            f"\nSelected number of clusters: "
            f"{selected_clusters}"
        )

        print(
            "Best silhouette score: "
            f"{best_row['silhouette_score']:.4f}"
        )
    else:
        selected_clusters = arguments.clusters

        if selected_clusters >= len(feature_table):
            raise ValueError(
                "The number of clusters must be smaller "
                "than the number of sampled routes."
            )

        evaluation_results = pd.DataFrame(
            [
                {
                    "number_of_clusters": selected_clusters,
                    "silhouette_score": np.nan,
                }
            ]
        )

        print(
            f"\nUsing the requested number of clusters: "
            f"{selected_clusters}"
        )

    final_model = KMeans(
        n_clusters=selected_clusters,
        random_state=arguments.random_seed,
        n_init=20,
    )

    cluster_labels = final_model.fit_predict(
        transformed_features
    )

    clustered_routes = feature_table.copy()
    clustered_routes["cluster"] = cluster_labels

    final_silhouette = silhouette_score(
        transformed_features,
        cluster_labels,
    )

    cluster_profiles = create_cluster_profiles(
        clustered_routes
    )

    assignments_path = (
        OUTPUT_DIRECTORY / "route_clusters.csv"
    )

    profiles_path = (
        OUTPUT_DIRECTORY / "cluster_profiles.csv"
    )

    evaluation_path = (
        OUTPUT_DIRECTORY / "cluster_evaluation.csv"
    )

    model_path = (
        MODEL_DIRECTORY / "kmeans_route_model.joblib"
    )

    clustered_routes.to_csv(
        assignments_path,
        index=False,
    )

    cluster_profiles.to_csv(
        profiles_path,
        index=False,
    )

    evaluation_results.to_csv(
        evaluation_path,
        index=False,
    )

    model_bundle = {
        "preprocessor": preprocessor,
        "kmeans_model": final_model,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "sample_size": len(feature_table),
        "random_seed": arguments.random_seed,
        "number_of_clusters": selected_clusters,
        "silhouette_score": final_silhouette,
    }

    joblib.dump(model_bundle, model_path)

    print("\nFinal cluster profiles:")
    print(cluster_profiles.to_string(index=False))

    print(
        f"\nFinal silhouette score: "
        f"{final_silhouette:.4f}"
    )

    print("\nThe application finished successfully.")

    print("\nFiles created:")
    print(f"Route features: {feature_table_path}")
    print(f"Route assignments: {assignments_path}")
    print(f"Cluster profiles: {profiles_path}")
    print(f"Model evaluation: {evaluation_path}")
    print(f"Saved model: {model_path}")


def main():
    """
    Starting point of the application.
    """
    arguments = parse_arguments()

    try:
        run_application(arguments)
    except Exception as error:
        print(f"\nThe application failed: {error}")
        raise


if __name__ == "__main__":
    main()