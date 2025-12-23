import pandas as pd
import pickle
import os

# Load CSV
df = pd.read_csv(r"D:\food_recommender\data\Food Ingredients and Recipe Dataset with Cleaned Ingredients.csv")

# Replace NaN
df = df.fillna("")

image_folder = r"D:\food_recommender\data\Food Images"

df["Image_Path"] = df["Image_Name"].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))

# Create embeddings folder if missing
os.makedirs(r"D:\food_recommender\embeddings", exist_ok=True)

# Build meta dictionary
meta = {
    "title": df["Title"].tolist(),
    "ingredients": df["Ingredients"].tolist(),
    "cleaned_ingredients": df["Cleaned_Ingredients"].tolist(),
    "instructions": df["Instructions"].tolist(),
    "image_path": df["Image_Path"].tolist(),
}

# Save meta.pkl
with open(r"D:\food_recommender\embeddings\meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("meta.pkl created successfully!")

