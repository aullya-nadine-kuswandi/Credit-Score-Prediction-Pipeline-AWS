from pathlib import Path
import pandas as pd


class DataIngestion:
    def __init__(self, input_path: str | Path, output_dir: str | Path):
        self.input_file = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "data_B.csv"  
        
    def run(self):
        print("--- Step 1: Data Ingestion ---")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.input_file)

        assert not df.empty, "Dataset is empty"
        assert "Credit_Score" in df.columns, "Target 'Credit_Score' tidak ditemukan"

        df.to_csv(self.output_file, index=False)
        print(f"✅ Data ingested from {self.input_file} → {self.output_file} | shape={df.shape}")

        return self.output_file
