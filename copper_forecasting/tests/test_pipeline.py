"""Tests for the copper_forecasting pipeline module."""

from copper_forecasting.models import GradientBoostingModel, LinearRegressionModel
from copper_forecasting.pipeline import ForecastingPipeline


class TestForecastingPipeline:
    """Tests for ForecastingPipeline."""

    def test_run_with_sample_data(self):
        pipeline = ForecastingPipeline(
            models=[
                LinearRegressionModel(),
                GradientBoostingModel(n_estimators=10, random_state=42),
            ],
        )
        results = pipeline.run()

        assert "model_results" in results
        assert "best_model" in results
        assert "best_metrics" in results
        assert "predictions" in results
        assert "test_actual" in results
        assert "feature_names" in results

    def test_best_model_selected(self):
        pipeline = ForecastingPipeline(
            models=[LinearRegressionModel()],
        )
        results = pipeline.run()
        assert results["best_model"] == "LinearRegression"

    def test_results_property(self):
        pipeline = ForecastingPipeline(
            models=[LinearRegressionModel()],
        )
        pipeline.run()
        assert pipeline.results == pipeline._results

    def test_predictions_shape_matches_test(self):
        pipeline = ForecastingPipeline(
            models=[LinearRegressionModel()],
        )
        results = pipeline.run()
        n_test = len(results["test_actual"])
        for model_name, preds in results["predictions"].items():
            assert len(preds) == n_test, f"Prediction length mismatch for {model_name}"

    def test_all_metrics_present(self):
        pipeline = ForecastingPipeline(
            models=[LinearRegressionModel()],
        )
        results = pipeline.run()
        metrics = results["best_metrics"]
        for key in ["MAE", "MSE", "RMSE", "MAPE", "R2", "DirectionalAccuracy"]:
            assert key in metrics

    def test_custom_split_ratios(self):
        pipeline = ForecastingPipeline(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            models=[LinearRegressionModel()],
        )
        results = pipeline.run()
        assert len(results["test_actual"]) > 0
