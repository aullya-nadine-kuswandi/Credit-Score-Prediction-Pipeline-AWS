import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, classification_report)

class ModelEvaluator:
    LABELS = ['Poor', 'Standard', 'Good']  # 0, 1, 2

    def run(self, train_results, x_test, y_test):
        print("--- Step 4: Evaluation (Test Set) ---")
        rows = {}

        for name, info in train_results.items():
            model = info['model']
            preds = model.predict(x_test)

            acc = accuracy_score(y_test, preds)
            f1m = f1_score(y_test, preds, average='macro')
            rec_poor = recall_score(y_test, preds, labels=[0], average='macro', zero_division=0)
            prec_good = precision_score(y_test, preds, labels=[2], average='macro', zero_division=0)

            rows[name] = {
                'cv_f1_macro': info['cv_f1_macro'],
                'test_accuracy': acc,
                'test_f1_macro': f1m,
                'test_recall_poor': rec_poor,
                'test_precision_good': prec_good,
            }

            print(f"\n[{name}] CV f1_macro={info['cv_f1_macro']:.4f} | "
                  f"TEST accuracy={acc:.4f} f1_macro={f1m:.4f} "
                  f"recall_poor={rec_poor:.4f} precision_good={prec_good:.4f}")
            print(classification_report(y_test, preds, target_names=self.LABELS))

        results_df = pd.DataFrame(rows).T
        best_name = max(rows, key=lambda k: rows[k]['cv_f1_macro'])
        best_model = train_results[best_name]['model']

        print(f"\nBest model (by CV f1_macro): {best_name} "
              f"(cv={rows[best_name]['cv_f1_macro']:.4f}, "
              f"test={rows[best_name]['test_f1_macro']:.4f})")
        return best_name, best_model, results_df
