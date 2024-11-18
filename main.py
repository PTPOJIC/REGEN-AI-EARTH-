import os
import subprocess

def main():
    print("Generating dataset...")
    subprocess.run(["python", "data/dataset_generator.py"])

    print("Training model...")
    subprocess.run(["python", "scripts/train.py"])

    print("Making predictions...")
    subprocess.run(["python", "scripts/predict.py"])

if __name__ == "__main__":
    main()
  
