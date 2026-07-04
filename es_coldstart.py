import pandas as pd

DATA_DIR = "data"

training_pairs = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")
submission_pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")

train_terms = set(training_pairs["term_id"].unique())
train_items = set(training_pairs["item_id"].unique())

sub_terms = set(submission_pairs["term_id"].unique())
sub_items = set(submission_pairs["item_id"].unique())

print("=== SUBMISSION PAIRS ===")
print(submission_pairs.shape)
print(submission_pairs.head())

print("\nUnique term (submission):", len(sub_terms))
print("Unique item (submission):", len(sub_items))

print("\n=== COLD-START ANALİZİ ===")
terms_seen = sub_terms & train_terms
terms_unseen = sub_terms - train_terms
print(f"Submission'daki terimlerden training'de görülenler: {len(terms_seen)} / {len(sub_terms)} (%{100*len(terms_seen)/len(sub_terms):.1f})")
print(f"Submission'daki terimlerden HİÇ görülmeyenler: {len(terms_unseen)} / {len(sub_terms)} (%{100*len(terms_unseen)/len(sub_terms):.1f})")

items_seen = sub_items & train_items
items_unseen = sub_items - train_items
print(f"\nSubmission'daki itemlardan training'de görülenler: {len(items_seen)} / {len(sub_items)} (%{100*len(items_seen)/len(sub_items):.1f})")
print(f"Submission'daki itemlardan HİÇ görülmeyenler: {len(items_unseen)} / {len(sub_items)} (%{100*len(items_unseen)/len(sub_items):.1f})")

# Satır bazında: kaç submission satırında term VE item ikisi de training'de görülmüş
both_seen_mask = submission_pairs["term_id"].isin(train_terms) & submission_pairs["item_id"].isin(train_items)
print(f"\nSubmission satırlarının term+item ikisi de training'de görülen oranı: %{100*both_seen_mask.mean():.2f}")
