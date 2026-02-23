import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("xauusd_train_pruned.csv")

# Optional: reduce columns
# df = df.iloc[:, :8]

sns.pairplot(df, diag_kind="kde")

plt.show()