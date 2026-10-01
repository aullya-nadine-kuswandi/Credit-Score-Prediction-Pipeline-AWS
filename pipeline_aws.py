import sys
from pathlib import Path
import joblib
from data_ingestion_aws import DataIngestion
from preprocessing_aws import CreditPreprocessor
from train_aws import CreditModelTrainer
from evaluation_aws import ModelEvaluator

class CreditScorePipeline:
    def __init__(self, raw_data_path, f1_threshold: float = 0.65,
                 recall_poor_threshold: float = 0.70,
                 precision_good_threshold: float = 0.58):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = self.base_dir / "ingested"
        self.artifact_dir = self.base_dir / "artifacts"
        self.f1_threshold = f1_threshold
        self.recall_poor_threshold = recall_poor_threshold
        self.precision_good_threshold = precision_good_threshold

        self.ingestor = DataIngestion(self.raw_data_path, self.ingested_dir)
        self.preprocessor = CreditPreprocessor()
        self.trainer = CreditModelTrainer()
        self.evaluator = ModelEvaluator()

    def execute(self):
        print("Executing Credit Score Pipeline (AWS, no MLflow)...\n")

        # Ingestion
        ingested_file = self.ingestor.run()

        # Preprocessing 
        x_train, x_test, y_train, y_test, transformer = self.preprocessor.run(ingested_file)

        # Training 3 model
        train_results = self.trainer.run(x_train, y_train, transformer)

        # Evaluation & pemilihan model terbaik berdasarkan hasil CV
        best_name, best_model, results_df = self.evaluator.run(train_results, x_test, y_test)

        print("\n--- Ringkasan Hasil ---")
        print(results_df[['cv_f1_macro', 'test_accuracy', 'test_f1_macro',
                          'test_recall_poor', 'test_precision_good']]
              .astype(float).round(4))

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.artifact_dir / "best_model.pkl"
        joblib.dump(best_model, best_path)
        print(f"\nBest model '{best_name}' saved -> {best_path}")

        best = results_df.loc[best_name]
        checks = [
            ('f1_macro',       float(best['test_f1_macro']),       self.f1_threshold),
            ('recall_poor',    float(best['test_recall_poor']),    self.recall_poor_threshold),
            ('precision_good', float(best['test_precision_good']), self.precision_good_threshold),
        ]
        print("\n--- Deployment Approval Decision ---")
        approved = True
        for name, value, thr in checks:
            passed = value >= thr
            approved = approved and passed
            print(f"  {name:14s}: {value:.4f}  (>= {thr})  [{'PASS' if passed else 'FAIL'}]")

        if approved:
            print(f"Approved: {best_name} lolos semua threshold -> siap deploy")
        else:
            print(f"Rejected: {best_name} tidak lolos semua threshold")


if __name__ == "__main__":
    DATA_INPUT = Path(__file__).parent / "data_B.csv"
    pipeline = CreditScorePipeline(raw_data_path=DATA_INPUT, f1_threshold=0.65)
    pipeline.execute()
