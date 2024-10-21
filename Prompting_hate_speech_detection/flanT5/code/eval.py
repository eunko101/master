import os
from datetime import datetime
import pandas as pd
import numpy as np
import time
import warnings
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    classification_report,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def format_output(output_filename):

    df = pd.read_csv(output_filename)
    if "pred" in df.columns:
        df["pred"] = df["pred"].str.lower()

    # weird exceptions
    df["pred"].replace(
        {
            "non.": "non",
            "oui.": "oui",
            "yes.": "yes",
            "no.": "no",
            "islamophobic": "yes",  # flan-large
            "sexist": "yes",  # flan-large
            "sexiste": "oui",  # flan-large
            "ableist": "oui",  # flan-large
            "agressif": "oui",  # flan-large
        },
        inplace=True,
    )

    if "b" or "d." or "c" in df["pred"].unique().tolist():
        df = df[~df["pred"].isin(["b", "d.", "c"])]

    # flan large returns the whole sentence
    df = df[df["pred"].str.len() <= 10]  # flan-large
    df = df[~df["pred"].isin([" super ", "d", "+1", "mt 1-1"])]  # flan-large

    values_labels = set(df["gold"].unique().tolist())
    values_labels_latin = set(["Sexiste", "NonSexiste", "Sexiste-reportage"])
    values_prediction = set(df["pred"].unique().tolist())

    set_binary_en = set(["yes", "no"])
    set_binary_fr = set(["oui", "non"])
    set_binary_mix = set_binary_en.union(set_binary_fr)

    if values_labels.issubset(set_binary_en):
        df["gold"].replace({"yes": 1, "no": 0}, inplace=True)
    elif values_labels.issubset(values_labels_latin):
        df["gold"].replace(
            {"Sexiste": 1, "NonSexiste": 0, "Sexiste-reportage": 0}, inplace=True
        )
    if (
        values_prediction.issubset(set_binary_en)
        or values_prediction.issubset(set_binary_fr)
        or values_prediction.issubset(set_binary_mix)
    ):
        df["pred"].replace({"yes": 1, "no": 0, "oui": 1, "non": 0}, inplace=True)
    return df


def evaluation_metrics(res_filename):
    # labels = [0, 1]
    res = format_output(res_filename)
    res = res[res.pred.notna()]
    print(res.head(10))

    print("gold_labels: ", list(set(res["gold"].unique().tolist())))
    print("pred_labels: ", list(set(res["pred"].unique().tolist())))

    print(list(set(res["gold"].unique().tolist() + res["pred"].unique().tolist())))
    # labels =[0,1]
    labels = list(set(res["gold"].unique().tolist() + res["pred"].unique().tolist()))

    if len(labels) > 1:
        tn, fp, fn, tp = confusion_matrix(
            res["gold"], res["pred"], labels=labels
        ).ravel()
    # fix this later
    elif len(labels) == 1:
        print(confusion_matrix(res["gold"], res["pred"], labels=labels).ravel())
        tn = 9999
        fp = 9999
        fn = 9999
        tp = 9999
    acc = accuracy_score(res["gold"], res["pred"])
    precision, recall, f1, support = precision_recall_fscore_support(
        res["gold"], res["pred"], labels=labels, zero_division=np.nan
    )

    macro = f1_score(
        res["gold"], res["pred"], average="macro", labels=labels, zero_division=np.nan
    )
    micro = f1_score(
        res["gold"], res["pred"], average="micro", labels=labels, zero_division=np.nan
    )
    weight = f1_score(
        res["gold"],
        res["pred"],
        average="weighted",
        labels=labels,
        zero_division=np.nan,
    )
    labels = list(map(str, labels))
    print(
        classification_report(
            res["gold"], res["pred"], target_names=labels, zero_division=np.nan
        )
    )
    return tn, fp, fn, tp, acc, precision, recall, f1, macro, micro, weight, support


def get_evaluation(dates):
    # note to self: shaden change the structure of folders to delete the one with the
    # model name
    directory = "./output/"
    filenames = os.listdir(directory)
    filenames_selected = []
    for date in dates:
        filenames_selected.extend(
            [filename for filename in filenames if filename.endswith(f"{date}.csv")]
        )
    # filenames = [filename for filename in filenames if filename.endswith(f"{date}.csv") for date in dates]

    df = pd.DataFrame(
        columns=[
            "filename",
            "prompt",
            "model",
            "dataset",
            "tn",
            "fp",
            "fn",
            "tp",
            "accuracy",
            "precision_class_0",
            "precision_class_1",
            "recall_class_0",
            "recall_class_1",
            "f1_class_0",
            "f1_class_1",
            "macro_f1",
            "micro_f1",
            "weighted_f1",
            "support",
        ]
    )

    for filename in filenames_selected:
        print(filename)
        tn, fp, fn, tp, acc, precision, recall, f1, macro, micro, weight, support = (
            evaluation_metrics(res_filename=os.path.join(directory, filename))
        )
        if len(precision) == 1:
            print(type(precision))
            precision = np.append(precision, 9999)
            print(precision)
            recall = np.append(recall, 9999)
            f1 = np.append(f1, 9999)
        print(precision)

        prompt = (
            filename.split("_")[4]
            + "_"
            + filename.split("_")[5]
            + "_"
            + filename.split("_")[6]
        )

        model = filename.split("_")[8]

        dataset = filename.split("_")[1]
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "filename": filename,
                            "prompt": prompt,
                            "model": model,
                            "dataset": dataset,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "tp": tp,
                            "accuracy": round(acc, 2),
                            "precision_class_0": round(precision[0], 2),
                            "precision_class_1": round(precision[1], 2),
                            "recall_class_0": round(recall[0], 2),
                            "recall_class_1": round(recall[1], 2),
                            "f1_class_0": round(f1[0], 2),
                            "f1_class_1": round(f1[1], 2),
                            "macro_f1": round(macro, 2),
                            "micro_f1": round(micro,2),
                            "weighted_f1": round(weight, 2),
                            "support": support,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    df = df.sort_values(by="prompt")
    timestr = time.strftime("%Y_%m_%d")
    df.to_csv(f"./output/evaluation_{timestr}_all.csv")


def main():
    dates = ["08_14", "08_15", "08_16"]
    get_evaluation(dates)


if __name__ == "__main__":
    main()
