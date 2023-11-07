"""
Visualize images from the dataset.
"""
import logging
from collections import Counter
from itertools import chain

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from HPAImageClassifier.HPAImageClassifier.config.config import DatasetConfig, LogsConfig
from HPAImageClassifier.HPAImageClassifier.config.constants import LABELS

sns.set_palette(sns.color_palette("rocket_r"))

data_df = pd.read_csv(DatasetConfig.TRAIN_CSV)
all_labels = list(chain.from_iterable([i.strip().split(" ") for i in data_df["Target"].values]))
c_val = Counter(all_labels)

n_keys = c_val.keys()
max_idx = max(n_keys)

counts = pd.DataFrame(
    {
        "Label": [LABELS[int(key)] for key in c_val.keys()],
        "Count": [val for val in c_val.values()],
    }
)

rev_label2id = {value: key for key, value in LABELS.items()}
counts["Class ID"] = [rev_label2id[label] for label in counts["Label"]]
counts = counts.set_index("Class ID")

counts = counts.sort_values(by="Count", ascending=False)
counts.style.background_gradient(cmap="Reds")

logging.info(f"Total number of images: {len(data_df)}")
logging.info(f"Total number of labels: {len(all_labels)}")
logging.info(f"{counts}")

plt.figure(figsize=(10, 5))
sns.barplot(y=counts["Label"].values, x=counts["Count"].values, order=counts["Label"].values)
plt.savefig(LogsConfig.VIZ_DIR + "/figs/Labels_barplot.png")
