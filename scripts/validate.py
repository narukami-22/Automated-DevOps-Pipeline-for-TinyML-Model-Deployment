import tensorflow as tf
import pandas as pd

model = tf.keras.models.load_model("models/model.h5")

df = pd.read_csv("data/clean_data.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

loss, acc = model.evaluate(X, y)

print("Accuracy:", acc)

if acc < 0.7:
    raise Exception("Model accuracy too low.")