import mlflow
from typing import Any


def create_mlflow_experiments(experiments_name : str , artifact_location : str , tags :dict[str,Any]) -> str : 
    try :
        experiment_id = mlflow.create_experiment(
        name = experiments_name,
        artifact_location = artifact_location,
        tags = tags,
        )
    except :
        print(f"Experiment {experiments_name} already exits ........")
        experiment_id = mlflow.get_experiment_by_name(experiments_name).experiment_id
    return experiment_id
    