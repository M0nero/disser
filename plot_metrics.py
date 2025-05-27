
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("data/epochs_metrics_v5.csv")

# График потерь
plt.figure(figsize=(10, 5))
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
plt.plot(df["Epoch"], df["Val Loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()

# График точности
plt.figure(figsize=(10, 5))
plt.plot(df["Epoch"], df["Train Acc"], label="Train Accuracy")
plt.plot(df["Epoch"], df["Val Acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()

# График F1-Score
plt.figure(figsize=(10, 5))
plt.plot(df["Epoch"], df["Val F1"], label="Validation F1 Score", color="purple")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Score over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("f1_score_plot.png")
plt.show()
