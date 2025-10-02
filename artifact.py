import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# 1. Create or load your dataset
data = {
    'feature_1': [1, 2, 3, 4, 5, 6, 7, 8],
    'target': [1.1, 1.9, 3.2, 4.0, 5.1, 5.8, 7.3, 7.9]
}
df = pd.DataFrame(data)

# Start an MLflow run
with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")

    # 2. Create a visualization
    fig, ax = plt.subplots()
    ax.scatter(df['feature_1'], df['target'])
    ax.set_title("Feature vs. Target")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Target")

    # 3. Log the figure as an artifact
    # This saves the plot to the run's artifacts in a 'visualizations' folder
    mlflow.log_figure(fig, "visualizations/scatter_plot.png")

    print("Plot has been logged to MLflow.")
