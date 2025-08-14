
# Classificação de Dígitos MNIST com PyTorch

Este projeto treina uma **rede neural convolucional (CNN)** para reconhecer dígitos escritos à mão do dataset MNIST. É uma aplicação clássica de **Deep Learning**.

## 🚀 Funcionalidades

- Treino de uma CNN no dataset MNIST
- Salvamento do melhor modelo (`best_model.pth`)
- Previsão de dígitos a partir de imagens externas (`predict.py`)
- Pipeline simples para demonstração de classificação de imagens

## 📦 Estrutura do Projeto

image-classification-starter/
│
├─ data/ # Dataset MNIST baixado automaticamente
├─ images/ # Imagens de teste (digitais feitas pelo usuário)
├─ outputs/ # Modelos treinados e meta.json
├─ train.py # Script de treino
├─ predict.py # Script de previsão
├─ requirements.txt # Dependências do projeto
└─ README.md # Este arquivo

## 🛠 Tecnologias

- Python 3
- PyTorch
- Torchvision
- Pillow
- tqdm

## ⚡ Instalação

1. Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd image-classification-starter


Crie e ative um ambiente virtual (opcional, mas recomendado):

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell


Instale as dependências:

pip install -r requirements.txt


Baixe o dataset MNIST (automaticamente pelo PyTorch):

python -c "from torchvision import datasets; datasets.MNIST(root='./data', train=True, download=True)"

🏋️‍♂️ Treinamento

Treine o modelo com:

python train.py --epochs 15 --batch_size 64


Isso salvará o melhor modelo em outputs/best_model.pth e o arquivo meta.json necessário para previsões.

🔮 Previsão

Para prever uma imagem externa:

python predict.py --image ./images/exemplo.png --model ./outputs/best_model.pth --out_dir ./outputs


Exemplo de saída:

Imagem: exemplo.png
Previsão: 5

📈 Resultados

Melhor acurácia: ~99.4%

Modelo robusto para dígitos manuscritos

💡 Observações

O arquivo meta.json contém informações sobre o dataset e classes e é necessário para o script de previsão.

Você pode adicionar imagens na pasta images/ para testar o modelo.

O projeto é totalmente funcional em CPU, mas se você tiver GPU, o PyTorch utilizará automaticamente.

📌 Conclusão

Este projeto demonstra todo o pipeline de classificação de imagens, desde o treino até a previsão, sendo um ótimo exemplo de Machine Learning e Deep Learning.
