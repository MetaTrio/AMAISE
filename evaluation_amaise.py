from helper import *
import pandas as pd
import logging
import click
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix

@click.command()
@click.option(
    "--pred",
    "-p",
    help="path to predicted labels file (txt format)",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--true",
    "-t",
    help="path to true labels file (csv format)",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output",
    "-o",
    help="path to output report file",
    type=click.Path(exists=False),
    required=True,
)
@click.help_option("--help", "-h", help="Show this message and exit")
def main(pred, true, output):

    # Set up logging
    logger = logging.getLogger(f"amaise")
    logger.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

   # Read the predicted labels from txt file (assuming comma-separated)
    pred_df = pd.read_csv(pred, sep=",", names=["id", "pred_label", "length"], header=0)

    # Read the true labels from CSV file with two columns: 'id' and 'y_true'
    true_df = pd.read_csv(true, usecols=['id', 'y_true'])

    # Dictionary to store predicted labels
    pred_dict = {}
    for _, row in pred_df.iterrows():
        pred_dict[row["id"]] = row["pred_label"]

    pred = []
    true = []

    # Build the predicted and true label lists
    for _, row in true_df.iterrows():
        true_label = row["y_true"]
        pred_label = pred_dict.get(row["id"])  # Get the predicted label (default to 0 if missing)

        # if pred host = 1 and pred microbe = 0 (This happens in orginal AMAISE test outputs) but true labels host is 0 and microbial is 1
        true.append(1-true_label)
       
        pred.append(pred_label)

    # Generate the classification report for binary classification
    report = classification_report(true, pred, target_names=["Microbial","Host"], output_dict=False)

     # Generate the confusion matrix
    conf_matrix = confusion_matrix(true, pred)

    # Format confusion matrix as text
    conf_matrix_df = pd.DataFrame(conf_matrix, index=["True Microbial", "True Host"], columns=["Predicted Microbial", "Predicted Host"])

    print("Confusion Matrix (Microbial vs. Host):")
    print(conf_matrix_df)

    print()

    print(report)


    # Output the report to a text file
    with open(output, 'w') as f:
        f.write(f"Binary Classification Report (Host vs. Microbial):\n\n")
        f.write(report)

        f.write("\nConfusion Matrix (Microbial vs. Host):\n")
        f.write(conf_matrix_df.to_string())  

    logger.info(f"Classification report and confusion matrix saved to: {output}")


if __name__ == "__main__":
    main()
