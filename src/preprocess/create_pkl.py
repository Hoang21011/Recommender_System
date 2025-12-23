import pandas as pd
import pickle
import os
image_folder = r"D:\project\cv_final\data\Food Images"
# Load CSV
df = pd.read_csv(r"D:\project\cv_final\data\Food Ingredients and Recipe Dataset with Image Name Mapping.csv")

df["Image_Path"] = df["Image_Name"].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))
# Replace NaN
df = df.fillna("")


# Create embeddings folder if missing
os.makedirs(r"D:\project\cv_final\embeddings", exist_ok=True)

# Build meta dictionary
meta = {
    "title": df["Title"].tolist(),
    "ingredients": df["Ingredients"].tolist(),
    "cleaned_ingredients": df["Cleaned_Ingredients"].tolist(),
    "instructions": df["Instructions"].tolist(),
    "image_path": df["Image_Path"].tolist(),
}

# Save meta.pkl
with open(r"D:\project\cv_final\embeddings\meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("meta.pkl created successfully!")