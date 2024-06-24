import mlflow
from typing import Any
import pandas as pd

from sklearn.datasets import make_classification

def create_mlflow_experiments(experiment_name : str , artifact_location : str , tags :dict[str,Any]) -> str : 
    try :
        experiment_id = mlflow.create_experiment(
        name = experiment_name,
        artifact_location = artifact_location,
        tags = tags,
        )
    except :
        print(f"Experiment {experiment_name} already exits ........")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    return experiment_id



def get_mlflow_experiment(experiment_id : str = None , experiment_name:str = None) -> mlflow.entities.Experiment:
    if experiment_id is not None :
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None :
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else :
        raise ValueError("Either experiment_id or experiment_name must be provided .")
    return experiment
    
    
def create_dataset(
    n_samples: int = 10000,
    n_features: int = 50,
    n_informative: int = 10,
    class_sep: float = 1.0,
) -> pd.DataFrame:
    """
    Create a dataset for testing purposes.

    :param n_samples: The number of samples.
    :param n_features: The number of features.
    :param n_informative: The number of informative features.
    :return: pd.DataFrame
    """

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        class_sep=class_sep,
        random_state=42,
    )

    df = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y

    return df