import os

print("Running Pipeline...")

os.system("python scripts/train.py")
os.system("python scripts/validate.py")
os.system("python scripts/convert.py")

print("Pipeline completed.")