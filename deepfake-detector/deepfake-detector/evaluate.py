"""Step 3: evaluate on the unseen test split -> metrics, confusion matrix, ROC, Grad-CAM samples."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from tensorflow import keras

import config
import gradcam
import inference
from model_utils import make_dataset


def gradcam_samples(grad_model, test_df, img_size, n_per_class=4):
    rows = pd.concat([test_df[test_df.label == c].sample(min(n_per_class, (test_df.label == c).sum()),
                                                         random_state=config.SEED) for c in (0, 1)])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, (_, r) in enumerate(rows.iterrows()):
        res = inference.analyze(grad_model, np.array(Image.open(r.path).convert("RGB")), img_size)
        row, col = i // 2, (i % 2) * 2
        axes[row, col].imshow(res["resized"])
        axes[row, col].set_title(f"True: {config.CLASS_NAMES[r.label]}", fontsize=9)
        axes[row, col + 1].imshow(res["overlay"])
        axes[row, col + 1].set_title(f"Pred: {res['label']} ({res['confidence']:.0%})", fontsize=9)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Grad-CAM: regions that influenced the prediction")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "gradcam_examples.png", dpi=130)
    plt.close(fig)


def run(model, test_df, img_size, batch_size=config.BATCH_SIZE):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    y_true = test_df["label"].values
    probs = model.predict(make_dataset(test_df, img_size, batch_size), verbose=1).ravel()
    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, probs)
    metrics = {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": auc,
               "test_images": int(len(y_true))}
    (config.OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4, zero_division=0)
    (config.OUTPUT_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig(config.OUTPUT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}"); plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
    plt.title("ROC curve"); plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "roc_curve.png", dpi=150); plt.close()

    gradcam_samples(gradcam.make_gradcam_model(model), test_df, img_size)
    print("\n" + report)
    print(json.dumps(metrics, indent=2))
    print(f"\nCharts saved in {config.OUTPUT_DIR}")


def main():
    meta = json.loads(config.META_PATH.read_text())
    model = keras.models.load_model(config.MODEL_PATH, compile=False)
    df = pd.read_csv(config.MANIFEST)
    run(model, df[df.split == "test"], meta["img_size"])


if __name__ == "__main__":
    main()
