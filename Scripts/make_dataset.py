import json


def clean_split_dataset():
    """Clean dataset, and split original val set into val_data and test_data"""
    
    # Read FinQA train and test data
    with open("Data/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open("Data/test.json", "r", encoding="utf-8") as f:
        org_val_data = json.load(f)

    # Apply the filter to remove non-numeric/blank answers
    print(f"Original train size: {len(train_data)}")
    train_data = filter_numeric_answers(train_data)
    print(f"Cleaned train size: {len(train_data)}")

    print(f"\nOriginal val/test size: {len(org_val_data)}")
    org_val_data = filter_numeric_answers(org_val_data)
    print(f"Cleaned val/test size: {len(org_val_data)}")

    # Use the first half strictly for the trainer's val_data
    midpoint = len(org_val_data) // 2
    val_data = org_val_data[:midpoint]

    # Use the rest strictly for final evaluate_model's test_data
    test_data = org_val_data[midpoint:] 

    print(f"Validation Set Size: {len(val_data)}")
    print(f"Test Set Size: {len(test_data)}")

    return train_data, val_data, test_data



def parse_ground_truth(ans):
    """Clean and convert a raw ground truth answer into a numeric float"""

    # Return None immediately if the input is null
    if ans is None:
        return None

    # Convert to string, strip whitespace, and remove commas
    ans = str(ans).strip().replace(",", "")
    
    # Return None if the string is empty after cleaning
    if ans == "":
        return None

    # Check for percentage sign and remove it
    is_percent = "%" in ans
    ans = ans.replace("%", "")

    # Attempt to convert to float, adjusting for percentages if necessary
    try:
        num = float(ans)
        if is_percent:
            num /= 100
        return num
    except:
        return None



def filter_numeric_answers(dataset):
    """Filter out dataset samples that have non-numeric or missing ground truth answers"""
    clean_data = []
    
    # Iterate through each sample in the dataset
    for sample in dataset:
        # Extract the raw ground truth answer from the sample
        ans = sample["qa"]["answer"]
        
        # Keep the sample only if the answer can be parsed into a valid number
        if parse_ground_truth(ans) is not None:
            clean_data.append(sample)
            
    return clean_data