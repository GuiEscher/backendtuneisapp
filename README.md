# 🏗️ API de Detecção de Defeitos com YOLOv8 (Flask)

Este projeto é uma API baseada em **Flask** que utiliza um modelo de Inteligência Artificial (**YOLOv8**) para detectar e segmentar defeitos estruturais em imagens e vídeos.

O sistema é capaz de identificar, classificar e gerar máscaras visuais para:
1.  **Umidade** (Verde)
2.  **Corrosão** (Azul)
3.  **Rachadura** (Vermelho)

---

## 📋 Funcionalidades

-   **Upload de Imagens:** Processa a imagem, desenha as caixas delimitadoras (bounding boxes) e aplica máscaras de segmentação nos defeitos.
-   **Upload de Vídeos:** Processa vídeos extraindo frames a cada 3 segundos, analisa-os e retorna um arquivo `.zip` contendo os frames processados.
-   **Retorno de Dados:** Retorna a imagem processada visualmente e os dados brutos (coordenadas, confiança, classe) via **Headers HTTP**.
-   **Câmera Local:** Rota para captura via webcam (apenas em execução local).

---

## 🛠️ Pré-requisitos

Para rodar este projeto, você precisará de:
-   Python 3.8 ou superior.
-   Arquivo de pesos do modelo treinado (`model.pt`) localizado na pasta `models/`.

---

## 🚀 Instalação e Execução Local

Siga os passos abaixo para testar em sua máquina:

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/GuiEscher/backendtuneisapp.git
    cd backendtuneisapp
    ```

2.  **Crie um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verifique o Modelo:**
    Certifique-se de que o arquivo `model.pt` está dentro da pasta `models`. A estrutura deve ser:
    ```
    /
    ├── models/
    │   └── model.pt
    ├── server.py
    ├── requirements.txt
    └── README.md
    ```

5.  **Inicie o Servidor:**
    ```bash
    python server.py
    ```
    O servidor iniciará (geralmente em `http://0.0.0.0:10000` ou `http://localhost:10000`).

---

## ☁️ Deploy no Render

O **Render** é uma plataforma de nuvem que permite hospedar esta API facilmente.

### ⚠️ Importante sobre o Acesso Web
Esta aplicação é uma **API REST**, não um site comum.
-   **Não acesse a URL raiz pelo navegador:** Se você abrir `https://sua-app.onrender.com` no Chrome/Edge, provavelmente verá um erro 404 ou 405. Isso é normal.
-   **Consumo:** A API foi desenhada para ser consumida por outros servidores, aplicativos mobile ou ferramentas de teste como Postman/Insomnia.

### Passo a Passo para Deploy

1.  **Crie sua conta:** Acesse [dashboard.render.com](https://dashboard.render.com/) e crie uma conta (é necessário ter a conta para gerenciar e acessar os logs).
2.  **Novo Web Service:**
    -   Clique em **"New +"** e selecione **"Web Service"**.
    -   Conecte seu repositório do GitHub/GitLab.
3.  **Configurações:**
    -   **Name:** Escolha um nome para seu serviço.
    -   **Runtime:** `Python 3`.
    -   **Build Command:** `pip install -r requirements.txt`
    -   **Start Command:** `python server.py`
4.  **Variáveis de Ambiente (Opcional):** O Render define a porta automaticamente na variável `PORT`, que o código já está preparado para ler.
5.  **Finalizar:** Clique em "Create Web Service".

> **Nota:** No plano gratuito (Free Tier), a API pode "dormir" após inatividade. A primeira requisição pode levar cerca de 50 segundos para "acordar" o servidor.

---

## 📡 Documentação da API

### 1. Verificar Status
Verifica se a API está online.
-   **Endpoint:** `GET /health`
-   **Resposta:** `200 OK` - `{"status": "healthy"}`

### 2. Detectar em Arquivo (Imagem ou Vídeo)
Este é o endpoint principal para consumo externo.

-   **Endpoint:** `POST /detect`
-   **Body (Multipart/Form-Data):**
    -   Key: `file`
    -   Value: (Arquivo de imagem .jpg/.png ou vídeo .mp4/.avi)

#### Exemplo de Resposta (Imagem):
-   **Body:** Retorna a imagem binária (JPEG) com as marcações desenhadas.
-   **Headers Especiais:**
    -   `Detections`: JSON contendo lista de objetos detectados (classe, confiança, bbox).
    -   `Logs`: Logs de processamento interno.

#### Exemplo de Resposta (Vídeo):
-   **Body:** Retorna um arquivo `.zip` contendo os frames analisados.
-   **Headers Especiais:**
    -   `Detections`: JSON com todas as detecções.
    -   `Frame-Count`: Quantidade de frames processados.

### 3. Captura Webcam (Apenas Local)
-   **Endpoint:** `GET /capture`
-   **Descrição:** Captura um frame instantâneo da webcam do servidor (não funciona no Render, pois servidores não têm webcams físicas).

---

## 💻 Exemplo de Consumo (Python Client)

Como os dados da detecção vêm no **Header** e a imagem no **Body**, veja como consumir a API (coloque a rota real do render esse é só um exemplo que fiz para facilitar o entendimento):

```python
import requests
import json

url = "[https://seu-app.onrender.com/detect](https://seu-app.onrender.com/detect)"
files = {'file': open('parede_rachada.jpg', 'rb')}

response = requests.post(url, files=files)

if response.status_code == 200:
    # 1. Recuperar os dados JSON dos cabeçalhos
    detections = json.loads(response.headers.get('Detections', '[]'))
    print("Defeitos encontrados:", detections)

    # 2. Salvar a imagem processada
    with open("resultado.jpg", "wb") as f:
        f.write(response.content)
    print("Imagem salva como resultado.jpg")
else:
    print("Erro:", response.text)
```

Para mais informações sobre o render: https://render.com/
Acesse também: https://render.com/docs
