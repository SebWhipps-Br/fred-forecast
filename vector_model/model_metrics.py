import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from autoregression_model import AutoRegressionModel
from static_model_evaluator import ModelEvaluator
from static_forecast import StaticFactorModel

def collect_metrics(true_values, fitted_or_predicted_values, model_name, variable_idx=None, baseline_values=None, evaluator=None):
    """
    Collect performance metrics for a given model and variable.

    Parameters:
    - true_values: np.array, actual data
    - fitted_or_predicted_values: np.array, fitted or predicted values
    - model_name: str, name of the model
    - variable_idx: int or None, index of the variable (None for multivariate)
    - baseline_values: np.array or None, baseline forecast values for DM test
    - evaluator: ModelEvaluator instance

    Returns:
    - dict, containing performance metrics
    """
    metrics = {'Model': model_name}
    if variable_idx is None:
        metrics['Variable'] = 'All'
        # Capture output of evaluate_fit or evaluate_performance
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        if 'Fit' in model_name:
            evaluator.evaluate_fit(true_values, fitted_or_predicted_values, model_name)
        else:
            evaluator.evaluate_performance(true_values, fitted_or_predicted_values, model_name, baseline_values)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Parse metrics from output
        for line in output.split('\n'):
            if 'MSE:' in line:
                metrics['MSE'] = float(line.split()[-1])
            elif 'RMSE:' in line:
                metrics['RMSE'] = float(line.split()[-1])
            elif 'MAE:' in line:
                metrics['MAE'] = float(line.split()[-1])
            elif 'MAPE (%):' in line:
                metrics['MAPE'] = float(line.split()[-1])
            elif 'R²:' in line:
                metrics['R²'] = float(line.split()[-1])
            elif 'Directional Accuracy (%):' in line:
                metrics['Directional Accuracy'] = float(line.split()[-1])
            elif 'DM Statistic:' in line:
                metrics['DM Statistic'] = float(line.split()[-1])
            elif 'P-value:' in line:
                metrics['DM P-value'] = float(line.split()[-1])
    else:
        variable_name = evaluator.model.df.columns[variable_idx]
        metrics['Variable'] = variable_name
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        if 'Fit' in model_name:
            evaluator.evaluate_fit(true_values, fitted_or_predicted_values, model_name, variable_idx)
        else:
            evaluator.evaluate_variable_performance(true_values, fitted_or_predicted_values, variable_idx, model_name, baseline_values)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Parse metrics from output
        for line in output.split('\n'):
            if 'MSE:' in line:
                metrics['MSE'] = float(line.split()[-1])
            elif 'RMSE:' in line:
                metrics['RMSE'] = float(line.split()[-1])
            elif 'MAE:' in line:
                metrics['MAE'] = float(line.split()[-1])
            elif 'MAPE (%):' in line:
                metrics['MAPE'] = float(line.split()[-1])
            elif 'R²:' in line:
                metrics['R²'] = float(line.split()[-1])
            elif 'Directional Accuracy (%):' in line:
                metrics['Directional Accuracy'] = float(line.split()[-1])
            elif 'DM Statistic:' in line:
                metrics['DM Statistic'] = float(line.split()[-1])
            elif 'P-value:' in line:
                metrics['DM P-value'] = float(line.split()[-1])
    return metrics

def main():
    # Configuration
    HORIZON = 12  # Forecast horizon in months
    VARIABLES = [0, 47, 71, 118]  # Variables to evaluate
    FILEPATH = 'preprocessed_current.csv'
    results = []

    # Load data
    df = pd.read_csv(FILEPATH, index_col=0, parse_dates=True)
    X = StandardScaler().fit_transform(df)

    # 1. AR(1) Baseline Model
    print("Running AR(1) Baseline Model...")
    ar_model = AutoRegressionModel(X, h=HORIZON)
    ar_model.fit()
    baseline_forecast = ar_model.forecasts
    fitted_values = ar_model.fitted_values
    baseline_evaluator = ModelEvaluator(StaticFactorModel(filepath=FILEPATH, q=7, h=HORIZON))

    # In-sample fit
    results.append(collect_metrics(ar_model.X_train, fitted_values, "AR(1) Fit", None, None, baseline_evaluator))
    for var in VARIABLES:
        results.append(collect_metrics(ar_model.X_train, fitted_values, "AR(1) Fit", var, None, baseline_evaluator))

    # Forecast performance
    results.append(collect_metrics(baseline_evaluator.model.X_actual, baseline_forecast, "AR(1) Forecast", None, None, baseline_evaluator))
    for var in VARIABLES:
        results.append(collect_metrics(baseline_evaluator.model.X_actual, baseline_forecast, "AR(1) Forecast", var, None, baseline_evaluator))

    # 2. Static Factor Model with Linear Regression
    print("\nRunning Static Factor Model (Linear)...")
    model_linear = StaticFactorModel(filepath=FILEPATH, q=7, h=HORIZON, pca_type='standard', forecast_method='linear')
    model_linear.run()
    evaluator_linear = ModelEvaluator(model_linear)

    # In-sample fit
    results.append(collect_metrics(model_linear.X_train, model_linear.common, "Static Linear Fit", None, None, evaluator_linear))
    for var in VARIABLES:
        results.append(collect_metrics(model_linear.X_train, model_linear.common, "Static Linear Fit", var, None, evaluator_linear))

    # Forecast performance
    results.append(collect_metrics(model_linear.X_actual, model_linear.forecast, "Static Linear Forecast", None, baseline_forecast, evaluator_linear))
    for var in VARIABLES:
        results.append(collect_metrics(model_linear.X_actual, model_linear.forecast, "Static Linear Forecast", var, baseline_forecast, evaluator_linear))

    # 3. Static Factor Model with Random Forest
    print("\nRunning Static Factor Model (Random Forest)...")
    model_rf = StaticFactorModel(filepath=FILEPATH, q=7, h=HORIZON, pca_type='standard', forecast_method='rf')
    model_rf.run()
    evaluator_rf = ModelEvaluator(model_rf)

    # In-sample fit
    results.append(collect_metrics(model_rf.X_train, model_rf.common, "Static RF Fit", None, None, evaluator_rf))
    for var in VARIABLES:
        results.append(collect_metrics(model_rf.X_train, model_rf.common, "Static RF Fit", var, None, evaluator_rf))

    # Forecast performance
    results.append(collect_metrics(model_rf.X_actual, model_rf.forecast, "Static RF Forecast", None, baseline_forecast, evaluator_rf))
    for var in VARIABLES:
        results.append(collect_metrics(model_rf.X_actual, model_rf.forecast, "Static RF Forecast", var, baseline_forecast, evaluator_rf))

    # Save results to DataFrame and CSV
    results_df = pd.DataFrame(results)
    # Reorder columns for clarity
    columns = ['Model', 'Variable', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R²', 'Directional Accuracy', 'DM Statistic', 'DM P-value']
    results_df = results_df[[col for col in columns if col in results_df.columns]]
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print("\nResults saved to 'model_evaluation_results.csv'")

if __name__ == "__main__":
    main()