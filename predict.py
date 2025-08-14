import argparse
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN
import json

def load_meta(out_dir):
    meta_path = Path(out_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Arquivo meta.json não encontrado em {out_dir}")
    with open(meta_path, "r") as f:
        return json.load(f)

def predict_image(image_path, model_path, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_meta(out_dir)
    classes = meta["classes"]

    # Transformações
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Carregar modelo
    model = SimpleCNN(in_channels=1, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Carregar imagem
    img = Image.open(image_path).convert("RGB")
    img_t = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        pred_idx = outputs.argmax(dim=1).item()
        pred_class = classes[pred_idx]

    print(f"Imagem: {image_path}")
    print(f"Previsão: {pred_class}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Caminho da imagem")
    parser.add_argument("--model", type=str, default="./outputs/best_model.pth")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    args = parser.parse_args()
    predict_image(args.image, args.model, args.out_dir)



