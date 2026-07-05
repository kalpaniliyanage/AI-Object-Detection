import clip
model, preprocess = clip.load("ViT-B/32", jit=False)
print("CLIP loaded fine")
