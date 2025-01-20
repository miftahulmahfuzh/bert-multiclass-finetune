import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_predictions(prediction_file, output_file):
    # Step 1: Read the TSV file and extract the relevant data
    # df = pd.read_csv(prediction_file, sep='\t') # miftah's output
    df = pd.read_csv(prediction_file)
    print(df)

    # Extract categories from <category></category> tags for both prediction and label columns
    # df['prediction_extracted'] = df['prediction'].str.extract(r'<category>(.*?)</category>') # miftah's output
    df['prediction_extracted'] = df['label_predicted'].str.extract(r'<category>(.*?)</category>') # legacy's output
    df['label_extracted'] = df['label'].str.extract(r'<category>(.*?)</category>')

    # Step 2: Check for novel categories in the 'prediction' column that don't exist in the 'label' column
    novel_categories = df[~df['prediction_extracted'].isin(df['label_extracted'])]

    if len(novel_categories) > 0:
        # List all novel categories with their respective row indices
        novel_categories_grouped = novel_categories.groupby('prediction_extracted').apply(lambda x: x.index.tolist())
        novel_categories_df = novel_categories_grouped.reset_index(name='row_indices')

        # Step 3: Check how many rows have novel categories
        print(f"Novel categories found: {novel_categories_grouped.shape[0]}")

        # Optionally print out the novel categories and the row indices
        print("Novel categories and their respective row indices:")
        print(novel_categories_df)

        # Count the number of rows with novel categories
        print(f"Novel categories found: {novel_categories.shape[0]}")

    # Step 4: Filter out rows with novel categories from the dataframe
    filtered_df = df[~df.index.isin(novel_categories.index)]
    filtered_df['original_index'] = filtered_df.index

    # Calculate accuracy and classification report
    y_true = filtered_df['label_extracted']
    y_pred = filtered_df['prediction_extracted']
    all_labels = set(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    if 'accuracy' not in report_df.index:
        report_df.loc['accuracy'] = accuracy
    else:
        report_df.at['accuracy', 'precision'] = accuracy
        report_df.at['accuracy', 'recall'] = accuracy
        report_df.at['accuracy', 'f1-score'] = accuracy
        report_df.at['accuracy', 'support'] = len(all_labels)


    # Save the results in an Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save the classification report as a dataframe
        report_df.to_excel(writer, sheet_name='Classification Report')

        # Optionally, save the filtered dataframe with novel categories removed
        filtered_df.to_excel(writer, sheet_name='Filtered Data', index=False)
        if len(novel_categories) > 0:
            novel_categories_df.to_excel(writer, sheet_name='Novel Categories', index=False)

    print(f"Evaluation results saved to {output_file}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report_df}")

# Example usage
# output_dir = "/home/devmiftahul/nlp/t5_dev/google/mt5-base_20250116_143532"
# i = 8
# prediction_file = f"{output_dir}/validation_predictions_epoch_{i}.tsv"  # Your input file
# output_file = f"{output_dir}/epoch_{i}_validation_result.xlsx"  # Output file for the results
output_dir = "/home/devmiftahul/nlp/legacy_t5/models/seq2seq/news_multiclass_no_pad-mt5-base-miftah-1/eval"
prediction_file = f"{output_dir}/predictions.csv"
output_file = f"{output_dir}/performance_scores.xlsx"
evaluate_predictions(prediction_file, output_file)
