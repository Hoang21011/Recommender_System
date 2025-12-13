import os
import pickle
import faiss
import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPTokenizer,CLIPModel

IMAGE_EMBED_PATH = r"D:\project\cv_final\embeddings\image_embeds_500.npy"
TEXT_EMBED_PATH = r"D:\project\cv_final\embeddings\text_embeddings_cleaned.npy"
META_PATH = r"D:\project\cv_final\embeddings\meta_clean_original_paths.pkl"
TEXT_FAISS_PATH = r"D:\project\cv_final\embeddings\faiss_text_index_cleaned.bin"
IMAGE_FAISS_PATH = r"D:\project\cv_final\embeddings\faiss_all_images.index"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_MODEL_NAME = "openai/clip-vit-large-patch14"

class FoodRetrievalSystem:
    def __init__(
        self,
        image_index_path=IMAGE_FAISS_PATH,
        image_embed_path=IMAGE_EMBED_PATH, #500 ảnh đầu
        text_index_path=TEXT_FAISS_PATH,
        text_embed_path=TEXT_EMBED_PATH,
        meta_path=META_PATH,
        model_name="openai/clip-vit-base-patch32",
        device=DEVICE
    ):
        self.device = device

        # Load FAISS index
        if not os.path.exists(image_index_path):
            raise FileNotFoundError(f"FAISS index not found: {image_index_path}")
        self.image_index = faiss.read_index(image_index_path)
        self.text_index = faiss.read_index(text_index_path)

        # Load image embeddings
        if not os.path.exists(image_embed_path):
            raise FileNotFoundError(f"Image embedding file not found: {image_embed_path}")
        self.image_embeds = np.load(image_embed_path)

        # Load metadata
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
        
        # Load CLIP model for queries
        # ----- Load CLIP models -----
        print("Loading TEXT model:", TEXT_MODEL_NAME)
        self.text_model = CLIPModel.from_pretrained(TEXT_MODEL_NAME).to(self.device)
        self.text_processor = CLIPProcessor.from_pretrained(TEXT_MODEL_NAME)

        print("Loading IMAGE model:", IMAGE_MODEL_NAME)
        self.image_model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME).to(self.device)
        self.image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)
    # -------------------------------------------------------------
    # Create embedding vector from uploaded image (query)
    # -------------------------------------------------------------
    def encode_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            emb = self.image_model.get_image_features(**inputs)

        emb = emb.cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # normalize
        return emb

    # -------------------------------------------------------------
    # Query bằng text ingredients (cleaned ingredients)
    # -------------------------------------------------------------
    def encode_text(self, text):
        inputs = self.text_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.text_model.get_text_features(**inputs)

        emb = emb.cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb
    # -------------------------------------------------------------
    # FAISS search function
    # -------------------------------------------------------------
    def search(self, query_emb, k=5, mode="text"):
        if mode == "text":
            distances, indices = self.text_index.search(query_emb, k)
        elif mode == "image":
            distances, indices = self.image_index.search(query_emb, k)
        else:
            raise ValueError("Mode must be 'text' or 'image'")

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            item = {
                "score": float(dist),
                "title": self.meta["title"][idx],
                "ingredients": self.meta["ingredients"][idx],
                "instructions": self.meta["instructions"][idx],
                "image_path": self.meta["image_path"][idx]

            }
            results.append(item)
        print("Query emb dim:", query_emb.shape)
        print("FAISS index dim:", self.image_index.d)
        print("Image_embeds dim:", self.image_embeds.shape)
        
        return results

    # -------------------------------------------------------------
    # Query bằng ảnh
    # -------------------------------------------------------------
    def search_by_image(self, image_path, k=5):
        query_emb = self.encode_image(image_path)
        return self.search(query_emb, k, mode="image")

    # -------------------------------------------------------------
    # Query bằng text (ingredient / tên món ăn / mô tả)
    # -------------------------------------------------------------
    def search_by_text(self, text, k=5):
        query_emb = self.encode_text(text)
        
        return self.search(query_emb, k, mode="text")


# -------------------------------------------------------------
# Main test
# -------------------------------------------------------------
if __name__ == "__main__":
    retrieval = FoodRetrievalSystem()

    # Example 1: Query by text
    print("Query: chicken tomato")
    results = retrieval.search_by_text("chicken tomato")
    for res in results:
        print(res)

    # Example 2: Query by image
    q_img = "D:\project\cv_final\data\Food-Images\-bloody-mary-tomato-toast-with-celery-and-horseradish-56389813.jpg"
    print("Query by image:", q_img)
    results = retrieval.search_by_image(q_img)
    for res in results:
        print(res)