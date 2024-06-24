from mlflow_utilities import create_mlflow_experiments
from mlflow import MlflowClient
if __name__=="__main__":

    experiment_id = create_mlflow_experiments(
        experiment_name="model_registry",
        artifact_location="model_registry_artifacts",
        tags={"purpose": "learning"},
    )

    print(experiment_id)

    client = MlflowClient()
    model_name = "registered_model_1"
    
    # create registered model
    # client.create_registered_model(model_name)

    # create model version 
    # source = ""
    # run_id = "da1d5bd925d94977af9247904b43cacd"
    # client.create_model_version(name=model_name, source=source, run_id=run_id)
    
    # transition model version stage 
    # client.transition_model_version_stage(name=model_name, version=1, stage="Archived")

    # delete model version 
    # client.delete_model_version(name=model_name, version=1)
    
    # delete registered model
    # client.delete_registered_model(name=model_name)