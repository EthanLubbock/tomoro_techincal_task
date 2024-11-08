import json
import pandas as pd

def preprocess_convfinqa_data(input_file, output_file):
    # Load the raw dataset from the input JSON file
    with open(input_file, 'r') as f:
        raw_data = json.load(f)

    # Initialize a list to store processed examples
    processed_data = []

    # Process each example in the dataset
    for example in raw_data:
        # Convert the table to a string format
        try:
            table_data = pd.DataFrame(
                example["table_ori"][1:],
                columns=example["table_ori"][0]
            )
            table_string = table_data.to_string(index=False)
        except ValueError:
            continue

        # Extract questions and answers using dialogue_break and exe_ans_list
        questions = example["annotation"]["dialogue_break"]
        answers = example["annotation"]["exe_ans_list"]

        # Create a dictionary with the processed data format
        processed_example = {
            "context": table_string,
            "questions": questions,
            "answers": answers
        }
        
        # Append to the processed data list
        processed_data.append(processed_example)

    # Save the processed data to an output JSON file
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)

if __name__ == "__main__":
    import os
    
    if not os.path.exists("data"):
        os.makedirs("data")

    input_paths = [
        "convfinqa_data/train.json",
        "convfinqa_data/dev.json"
    ]
    output_paths = [
        "data/train.json",
        "data/validation.json"
    ]
    for input_file, output_file in zip(
        input_paths, output_paths
    ):
        print(f"Processing: {input_file}")
        preprocess_convfinqa_data(input_file, output_file)