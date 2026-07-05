import sys
import pandas as pd

DATA_DIR = "data"

if len(sys.argv) < 2:
    print("Kullanım: python generate_single_threshold.py <threshold>")
    sys.exit(1)

threshold = float(sys.argv[1])

prob_df = pd.read_csv(f"{DATA_DIR}/submission_probs.csv")
sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

preds = (prob_df["prob"].values >= threshold).astype(int)
positive_rate = preds.mean()

submission = pd.DataFrame({"id": prob_df["id"], "prediction": preds})
submission = submission.set_index("id").loc[sample["id"]].reset_index()

out_path = f"{DATA_DIR}/submission_threshold_{threshold:.2f}.csv"
submission.to_csv(out_path, index=False)
print(f"threshold={threshold:.2f}  relevant_oranı=%{100*positive_rate:.1f}  -> {out_path}")
