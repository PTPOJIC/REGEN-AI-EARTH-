import os
import subprocess

def main():
    print("Generating dataset...")
    subprocess.run(["python3", "data/dataset_generator.py"])

    print("Training model...")
    subprocess.run(["python3", "scripts/train.py"])

    print("Making predictions...")
    subprocess.run(["python3", "scripts/predict.py"])

if __name__ == "__main__":
    main()
  
