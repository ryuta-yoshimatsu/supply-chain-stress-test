# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks notebook source
# MAGIC
# MAGIC Separate business logic into specific markdown sections (ensure space after pound sign)

# COMMAND ----------

import ray
import mlflow
import uuid
import pandas as pd
import os

# Start Ray (on cluster driver, workers will connect automatically)
if not ray.is_initialized():
    ray.init(address="auto")
    
@ray.remote
def run_and_log_ray(scenario, batch_id):
    mlflow.set_experiment("/SupplyChainStressTest")
    with mlflow.start_run(run_name=scenario["name"]):
        mlflow.set_tag("batch_id", batch_id)
        mlflow.log_param("scenario_name", scenario["name"])
        mlflow.log_param("factory_down", scenario.get("factory_down", "None"))
        results = run_scenario(
            scenario_name=scenario["name"],
            factory_down=scenario.get("factory_down")
        )
        mlflow.log_metric("total_cost", results["cost"])
        mlflow.log_metric("total_unmet_demand", results["unmet"])
        # Log shipping plan
        df_ship = results.get("shipping_plan_df")
        if df_ship is not None:
            output_path = f"/tmp/shipping_plan_{scenario['name']}.csv"
            df_ship.to_csv(output_path, index=False)
            mlflow.log_artifact(output_path)
            os.remove(output_path)
        return results
scenario_list = [
    {"name": "Baseline", "factory_down": None},
    {"name": "F1_Down", "factory_down": "F1"},
    {"name": "F2_Down", "factory_down": "F2"},
    # Add more for scaling...
]
batch_id = str(uuid.uuid4())
# Dispatch all tasks
futures = [run_and_log_ray.remote(s, batch_id) for s in scenario_list]
results = ray.get(futures)  # Gather results
# To DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("ray_scenario_results.csv", index=False)
results_df.head()
