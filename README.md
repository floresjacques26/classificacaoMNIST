# 🖼️ Image Classification Starter (MNIST / CIFAR-10)

Projeto simples em **PyTorch** para treinar um classificador de imagens usando **MNIST** (mais fácil) ou **CIFAR‑10** (um pouco mais desafiador). O código já faz **download automático** do dataset via `torchvision`.

## ✅ Requisitos
- Python 3.9+
- Pip atualizado

## 🚀 Passo a passo rápido

```bash
# 1) Crie e ative um ambiente virtual (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Instale dependências
pip install -r requirements.txt

# 3) Treine no MNIST (recomendado para começar)
python train.py --dataset mnist --epochs 5

# 4) Ou treine no CIFAR-10
python train.py --dataset cifar10 --epochs 10

# 5) (Opcional) Defina diretórios
python train.py --dataset mnist --data_dir ./data --out_dir ./outputs
```

Os dados serão baixados automaticamente para `--data_dir` na primeira execução.

## 🧠 O que o script faz?
- Faz download do dataset (MNIST ou CIFAR-10)
- Cria **DataLoaders** com transforms adequadas e normalização
- Define uma **CNN simples**
- Treina e avalia **acurácia** no conjunto de teste
- Salva o melhor modelo em `outputs/best_model.pth` + `outputs/meta.json`

## 📁 Estrutura
```
image-classification-starter/
├─ train.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## 🧪 Exemplos de uso
```bash
# Treinar com batch maior e por mais épocas
python train.py --dataset mnist --epochs 8 --batch_size 128 --lr 1e-3
```

## ✨ Dicas
- Comece com **MNIST** (rápido e estável). Depois migre para **CIFAR‑10**.
- Se tiver GPU (CUDA), o script usa automaticamente.
- Para ver algumas amostras, use `--show_samples 1` (abre uma janela matplotlib).
```bash
python train.py --dataset mnist --show_samples 1
```

Bom estudo e bons treinos! 🚀