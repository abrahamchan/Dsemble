import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix


def filter_label_noise_transition_matrix(df):
    df.loc[(df["is_label_issue"] == False) & (df["label_score"] > 0.0001), "predicted_label"] = df["given_label"]
    return df


def generate_heapmap(transition_df):
    sn.set(font_scale=2.0) # for label size
    sn.heatmap(transition_df, annot=False, annot_kws={"size": 20}, fmt=".0f", cmap="YlGnBu",
               xticklabels=transition_df.columns, yticklabels=transition_df.columns,
               )

    plt.xlabel("Given Labels", fontsize=18)
    plt.ylabel("Cleanlab True Labels", fontsize=18)
    plt.yticks(rotation=0)

    plt.show()


def main(argv):
    label_issues_file = argv[0]
    output_file = argv[1]

    df = pd.read_csv(label_issues_file)

    columns = df["given_label"].drop_duplicates().sort_values()
    df = filter_label_noise_transition_matrix(df)

    cm = confusion_matrix(df["predicted_label"], df["given_label"])

    transition_df = pd.DataFrame.from_dict(dict(zip(columns, zip(*cm))))
    transition_df.to_csv(output_file)

    generate_heapmap(transition_df)


if __name__ == "__main__":
    main(sys.argv[1:])

