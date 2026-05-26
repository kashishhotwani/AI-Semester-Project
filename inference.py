
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    model = load_model()
    with torch.no_grad():
        output = model(input_tensor)
        predicted = classes[output.argmax(1).item()]
    return predicted

if __name__ == "__main__":
    test_folder = "/content/dataset/seg_test/seg_test"
    sample_images = []
    for class_name in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_name)
        if os.path.isdir(class_path):
            img_file = os.listdir(class_path)[0]
            sample_images.append(os.path.join(class_path, img_file))
        if len(sample_images) == 4:
            break
    for img_path in sample_images:
        print(f"Image: {img_path} → Predicted: {predict(img_path)}")
