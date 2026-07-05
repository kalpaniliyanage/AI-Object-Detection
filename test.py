import clip
model, preprocess = clip.load("ViT-B/32", jit=False)
print("CLIP loaded fine")

from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-world.pt")

print("YOLO-World model loaded successfully!")

classes = [
    "person",
    "book",
    "bottle",
    "cup",
    "cell phone",
]

model.set_classes(classes)

print("Classes loaded:")
for item in classes:
    print("-", item)
