from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import re
import numpy as np

app = Flask(__name__)

# CRNN Model
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 27)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, 64)
        x, _ = self.rnn(x)
        return self.fc(x[:, -1, :])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN().to(device)
model.load_state_dict(torch.load("crnn_emnist_letters.pth", map_location=device))
model.eval()

labels = [chr(i + 64) for i in range(1, 27)]  # A-Z

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.json['image']
    img_str = re.search(r'base64,(.*)', data_url).group(1)
    img_bytes = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_bytes)).convert('L')
    image = Image.fromarray(255 - np.array(image))  # invert white-on-black

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        pred = out.argmax(1).item()

    if 1 <= pred <= 26:
        result = labels[pred - 1]
    else:
        result = "?"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
