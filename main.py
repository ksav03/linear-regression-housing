from src.data_loader import save_dataset_as_csv
from src.train import train_and_evaluate

def main():
    save_dataset_as_csv()

    train_and_evaluate()


if __name__ == "__main__":
    main()