import numpy as np, faiss, pickle, os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# paths (chỉnh nếu cần)
IMAGE_EMBED_PATH = r"D:\project\cv_final\embeddings\image_embeds_500.npy"
IMAGE_FAISS_PATH  = r"D:\project\cv_final\embeddings\faiss_image_index_cleaned.bin"
MODEL_NAME        = "openai/clip-vit-base-patch32"   # model bạn đang load để encode query

print("Files exist:",
      os.path.exists(IMAGE_EMBED_PATH),
      os.path.exists(IMAGE_FAISS_PATH))

# load npy & index & model (model load chỉ để kiểm tra dim of query)
embs = np.load(IMAGE_EMBED_PATH)
print("image_embeds shape:", embs.shape)

idx = faiss.read_index(IMAGE_FAISS_PATH)
print("faiss index dim:", idx.d, "ntotal:", idx.ntotal)

# load CLIP model (to check what query dim would be)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained(MODEL_NAME, local_files_only=False)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
# get text/image projection dim from model config if possible:
cfg_dim = getattr(model.config, "projection_dim", None) or getattr(model.config, "text_config", {}).get("projection_dim", None)
print("model config projection_dim:", cfg_dim)
# build a sample query emb shape
from PIL import Image
img = Image.open(r"D:\project\cv_final\data\Food-Images\-bloody-mary-tomato-toast-with-celery-and-horseradish-56389813.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    q = model.get_image_features(**inputs).cpu().numpy()
print("sample query emb shape:", q.shape)
