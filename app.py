
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

app = Flask(__name__)

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Load model once
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Image Classification API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted = classes[output.argmax(1).item()]
    
    return jsonify({"prediction": predicted})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
