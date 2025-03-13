import pandas as pd
from datasets import load_dataset

# Load a sample financial dataset (SEC filings, earnings reports)
dataset = load_dataset("gretelai/gretel-financial-risk-analysis-v1")

# Convert to DataFrame
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

# Save locally
df_train.to_csv("data/train_financial.csv", index=False)
df_test.to_csv("data/test_financial.csv", index=False)

print("✅ Financial data loaded and saved locally!")
print(df_train[0])
