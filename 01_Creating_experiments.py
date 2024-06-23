from mlflow_utilities import create_mlflow_experiments


if __name__ == "__main__" :
    # mlflow.create_experiment(
    #     name = "test_mlflow1",
    #     artifact_location = "test_mlflow1_artifacts",
    #     tags = {"env" : "dev" , "version" : "1.0.0"},
    # )
    
    experiment_id = create_mlflow_experiments("test_mlflow2","test_mlflow1_artifacts",{"env" : "dev" , "version" : "1.0.0"})
    print(f"Expirement ID : {experiment_id} ")