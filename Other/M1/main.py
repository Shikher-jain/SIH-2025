import subprocess


def main():
    # Run each script in sequence
    print(1)
    subprocess.run(["python", "1_preprocess_data.py"])
    print(2)
    subprocess.run(["python", "3_train_cnn_model.py"])
    print(3)
    subprocess.run(["python", "4_evaluate_model.py"])

if __name__ == "__main__":
    main()