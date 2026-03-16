from Scripts.make_dataset import clean_split_dataset
from Scripts.evaluation import clear_vram, load_model_for_evaluation, evaluate_model
from Scripts.model import prepare_trainer, finetune_model



def main():
    """Main function to run the entire pipeline for fine-tuning and evaluation of LLM model to perform mathematical reasoning on financial questions"""
    # Step A: Clean dataset, and split original val set into val_data and test_data
    print("\nStep A: Clean dataset, and split original val set into val_data and test_data\n")
    train_data, val_data, test_data = clean_split_dataset()

    # Step B1: Load the model BEFORE fine-tuning
    print("\nStep B1: Load the model BEFORE fine-tuning\n")
    clear_vram()
    tokenizer1, model1 = load_model_for_evaluation(load_peft = False)

    # Step B2: Evaluate accuracy of the model BEFORE fine-tuning
    print("\nStep B2: Evaluate accuracy of the model BEFORE fine-tuning\n")
    evaluate_model(test_data, tokenizer1, model1, n=547)

    # Step C1: Prepare the model trainer/settings
    print("\nStep C1: Prepare the model trainer/settings\n")
    tokenizer, model, trainer = prepare_trainer(train_data, val_data)

    # Step C2: Fine tune the model
    print("\nStep C2: Fine tune the model\n")
    finetune_model(tokenizer, model, trainer)

    # Step D1: Load the model AFTER fine-tuning
    print("\nStep D1: Load the model AFTER fine-tuning\n")
    clear_vram()
    tokenizer2, model2 = load_model_for_evaluation(load_peft = True)

    # Step D2: Evaluate accuracy of the model AFTER fine-tuning
    print("\nStep D2: Evaluate accuracy of the model AFTER fine-tuning\n")
    evaluate_model(test_data, tokenizer2, model2, n=547)



if __name__ == "__main__":
    main()