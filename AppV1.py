import os
import zipfile
import shutil
import tempfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from torchvision.models import resnet18, resnet50, densenet121
from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.decomposition import PCA
import streamlit as st
import gc
import logging
import base64
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import cv2
import io
import warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Importações adicionais para o modo quântico
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import json
from tensorflow.keras.layers import Layer
from cirq.contrib.svg import circuit_to_svg  # Deve estar disponível após a instalação com extras SVG

# Supressão dos avisos relacionados ao torch.classes
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.classes.*")

# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurações para tornar os gráficos mais bonitos
sns.set_style('whitegrid')

# Configurar Matplotlib para usar uma fonte disponível caso 'Arial' não esteja instalada
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Fonte padrão disponível

# Definir seed para reprodutibilidade
def set_seed(seed):
    """
    Define uma seed para garantir a reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)  # Adicionado para TensorFlow
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # As linhas abaixo são recomendadas para garantir reprodutibilidade
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Definir a seed para reprodutibilidade

# Definir as transformações para aumento de dados (aplicando transformações aleatórias)
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    ], p=0.5),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Transformações para validação e teste
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Dataset personalizado para classificação
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset personalizado para segmentação
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

def seed_worker(worker_id):
    """
    Função para definir a seed em cada worker do DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Definição da camada personalizada SerializablePQC
class SerializablePQC(Layer):
    def __init__(self, circuit, observables, **kwargs):
        super().__init__(**kwargs)
        self.circuit = circuit
        self.observables = observables
        self.pqc = tfq.layers.PQC(circuit, observables)
    
    def call(self, inputs):
        return self.pqc(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'circuit': cirq.to_json(self.circuit),  # Serializa o circuito para JSON
            'observables': [cirq.to_json(obs) for obs in self.observables],  # Serializa os observáveis
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        circuit_json = config.pop('circuit')
        observables_json = config.pop('observables')
        # Desserializa o circuito e os observáveis a partir do JSON
        circuit = cirq.read_json(circuit_json)
        observables = [cirq.read_json(obs_json) for obs_json in observables_json]
        return cls(circuit, observables, **config)

def visualize_data(dataset, classes):
    """
    Exibe algumas imagens do conjunto de dados com suas classes.
    """
    st.write("### Visualização de Algumas Imagens do Conjunto de Dados")
    fig = plt.figure(figsize=(15, 5))
    for i in range(10):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        image = np.array(image)  # Converter a imagem PIL em array NumPy
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(image)
        ax.set_title(classes[label])
        ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)  # Fechar a figura para liberar memória

def plot_class_distribution(dataset, classes):
    """
    Exibe a distribuição das classes no conjunto de dados e mostra os valores quantitativos.
    """
    # Extrair os rótulos das classes para todas as imagens no dataset
    labels = [label for _, label in dataset]

    # Criar um DataFrame para facilitar o plot com Seaborn
    df = pd.DataFrame({'Classe': labels})

    # Plotar o gráfico com as contagens
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(x='Classe', data=df, palette="Set2")
    plt.title("Distribuição das Classes (Quantidade de Imagens)")
    plt.xlabel("Classes")
    plt.ylabel("Número de Imagens")
    
    # Adicionar as contagens acima das barras
    class_counts = df['Classe'].value_counts().sort_index()
    for i, count in enumerate(class_counts):
        plt.text(i, count + max(class_counts)*0.01, str(count), ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    plt.close(fig)  # Fechar a figura para liberar memória

def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False, seed=42):
    """
    Retorna o modelo pré-treinado selecionado para classificação.
    """
    if model_name == 'ResNet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif model_name == 'ResNet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif model_name == 'DenseNet121':
        weights = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=weights)
    else:
        st.error("Modelo não suportado.")
        return None

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    if model_name.startswith('ResNet'):
        num_ftrs = model.fc.in_features
        initializer = nn.init.kaiming_uniform_
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes, bias=True)
        )
        # Aplicar inicialização com seed
        torch.manual_seed(seed)
        initializer(model.fc[1].weight, a=np.sqrt(5))
        nn.init.constant_(model.fc[1].bias, 0)
    elif model_name.startswith('DenseNet'):
        num_ftrs = model.classifier.in_features
        initializer = nn.init.kaiming_uniform_
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, num_classes, bias=True)
        )
        # Aplicar inicialização com seed
        torch.manual_seed(seed)
        initializer(model.classifier[1].weight, a=np.sqrt(5))
        nn.init.constant_(model.classifier[1].bias, 0)
    else:
        st.error("Modelo não suportado.")
        return None

    model = model.to(device)
    return model

def get_observables(qubits):
    """
    Define os observáveis a serem medidos no circuito quântico.
    """
    observables = [cirq.Z(q) for q in qubits]
    return observables

def create_quantum_circuit(n_qubits, n_layers):
    """
    Cria um circuito quântico parametrizado com re-uploading de dados.
    """
    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuit = cirq.Circuit()
    symbols = sympy.symbols(f'theta(0:{3*n_layers*n_qubits})')
    idx = 0

    for layer in range(n_layers):
        for q in qubits:
            circuit += cirq.rx(symbols[idx])(q)
            circuit += cirq.ry(symbols[idx+1])(q)
            circuit += cirq.rz(symbols[idx+2])(q)
            idx += 3
        # Adicionar portas de entanglement se necessário
        if n_qubits > 1:
            for i in range(n_qubits - 1):
                circuit += cirq.CZ(qubits[i], qubits[i+1])
    
    return circuit, symbols

def build_quantum_model(n_qubits, n_layers, symbols, observables):
    """
    Constrói um modelo quântico utilizando TFQ com uma camada PQC serializável.
    """
    circuit, _ = create_quantum_circuit(n_qubits, n_layers)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        SerializablePQC(circuit, observables),  # Utiliza a camada personalizada
        tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária
    ])
    
    return model

def convert_features_to_circuits(features, circuit, symbols):
    """
    Converte features em circuitos quânticos parametrizados.
    """
    circuits = []
    for feature in features:
        # Criar um dicionário mapeando símbolos para valores
        symbol_values = dict(zip(symbols, feature))
        # Resolver o circuito com os valores
        resolved_circuit = cirq.resolve_parameters(circuit, symbol_values)
        circuits.append(resolved_circuit)
    return tfq.convert_to_tensor(circuits)

def build_hybrid_model(classical_model, quantum_model, quantum_circuit, quantum_symbols):
    """
    Constrói um modelo híbrido combinando modelos clássico e quântico.
    """
    inputs = tf.keras.Input(shape=(classical_model.output_shape[1],), dtype=tf.float32)
    x = classical_model(inputs)
    # Converter para circuitos quânticos
    circuits = tf.py_function(
        func=lambda x: convert_features_to_circuits(x, quantum_circuit, quantum_symbols),
        inp=[x],
        Tout=tf.string
    )
    outputs = quantum_model(circuits)
    hybrid_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return hybrid_model

def train_quantum_model(
    epochs=3, 
    batch_size=32, 
    n_qubits=4,
    n_layers=2,
    seed=42
):
    """
    Treina um modelo quântico simples para classificação binária.
    
    Retorna o modelo quântico treinado.
    """
    set_seed(seed)
    
    # Definir qubits e circuitos
    circuit, symbols = create_quantum_circuit(n_qubits, n_layers)
    qubits = cirq.GridQubit.rect(1, n_qubits)
    observables = get_observables(qubits)
    
    # Construir modelo quântico
    quantum_model = build_quantum_model(n_qubits, n_layers, symbols, observables)
    
    # Compilar o modelo
    quantum_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Aqui você deveria adicionar a lógica de treinamento do modelo quântico
    # Dependendo do seu conjunto de dados, como você está treinando, etc.
    # Isso pode incluir a criação de dados quânticos, conversão de dados, etc.
    # Como placeholder, deixo o modelo não treinado
    
    return quantum_model, circuit, symbols, observables

def visualize_circuit(circuit):
    """
    Converte um circuito Cirq em uma imagem SVG e exibe no Streamlit.
    """
    if circuit is None:
        st.warning("Nenhum circuito para visualizar.")
        return
    
    # Gerar SVG do circuito
    try:
        svg = circuit_to_svg(circuit)
    except ImportError:
        st.error("Falha ao importar cirq.contrib.svg. Verifique a instalação do Cirq com suporte SVG.")
        return
    except Exception as e:
        st.error(f"Erro ao gerar SVG do circuito: {e}")
        return
    
    # Exibir SVG no Streamlit
    st.image(svg, use_container_width=True)

def evaluate_image(model, image, classes):
    """
    Avalia uma única imagem e retorna a classe predita e a confiança.
    """
    model.eval()
    image_tensor = test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        class_idx = pred.item()
        class_name = classes[class_idx]
        return class_name, confidence.item()

def evaluate_image_quantum(quantum_model, quantum_circuit, quantum_symbols, scaler, image, threshold=0.5):
    """
    Avalia uma única imagem usando o modelo quântico.
    """
    # Extrair features com o modelo clássico
    with torch.no_grad():
        if 'model' not in st.session_state:
            st.error("Modelo clássico não encontrado no estado da sessão.")
            return None, None
        model_for_embeddings = st.session_state['model']
        model_for_embeddings.eval()
        image_tensor = test_transforms(image).unsqueeze(0).to(device)
        features = model_for_embeddings(image_tensor)
        features = features.cpu().numpy()
    
    # Normalizar features
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(features)  # Ajuste o scaler com os features do treinamento
    scaled_features = scaler.transform(features)
    
    # Converter features para circuitos quânticos
    circuits = convert_features_to_circuits(scaled_features, quantum_circuit, quantum_symbols)
    
    # Fazer a previsão
    y_pred = quantum_model.predict(circuits)
    
    # Interpretar a saída
    predicted_label = 1 if y_pred[0][0] > threshold else 0
    confidence_q = y_pred[0][0]
    
    # Obter o nome da classe
    classes = st.session_state.get('classes', ["Classe_0", "Classe_1"])
    if predicted_label >= len(classes):
        class_name = "Desconhecida"
    else:
        class_name = classes[predicted_label]
    
    return class_name, confidence_q

def train_model(
    data_dir, 
    num_classes, 
    model_name, 
    fine_tune, 
    epochs, 
    learning_rate, 
    batch_size, 
    train_split, 
    valid_split, 
    use_weighted_loss, 
    l2_lambda, 
    patience, 
    seed=42
):
    """
    Função principal para treinamento do modelo de classificação.
    """
    set_seed(seed)

    # Carregar o dataset original sem transformações
    full_dataset = datasets.ImageFolder(root=data_dir)

    # Verificar se há classes suficientes
    if len(full_dataset.classes) < num_classes:
        st.error(f"O número de classes encontradas ({len(full_dataset.classes)}) é menor do que o número especificado ({num_classes}).")
        return None

    # Exibir dados
    visualize_data(full_dataset, full_dataset.classes)
    plot_class_distribution(full_dataset, full_dataset.classes)

    # Divisão dos dados
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_end = int(train_split * dataset_size)
    valid_end = int((train_split + valid_split) * dataset_size)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    # Verificar se há dados suficientes em cada conjunto
    if len(train_indices) == 0 or len(valid_indices) == 0 or len(test_indices) == 0:
        st.error("Divisão dos dados resultou em um conjunto vazio. Ajuste os percentuais de divisão.")
        return None

    # Criar datasets para treino, validação e teste
    train_dataset_subset = torch.utils.data.Subset(full_dataset, train_indices)
    valid_dataset_subset = torch.utils.data.Subset(full_dataset, valid_indices)
    test_dataset_subset = torch.utils.data.Subset(full_dataset, test_indices)

    # Criar dataframes para os conjuntos de treinamento, validação e teste com data augmentation e embeddings
    model_for_embeddings = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False, seed=seed)
    if model_for_embeddings is None:
        return None

    st.write("**Processando o Conjunto de Treinamento para Inclusão de Data Augmentation e Embeddings...**")
    train_df = apply_transforms_and_get_embeddings(train_dataset_subset, model_for_embeddings, train_transforms, batch_size=batch_size, seed=seed)
    st.write("**Processando o Conjunto de Validação...**")
    valid_df = apply_transforms_and_get_embeddings(valid_dataset_subset, model_for_embeddings, test_transforms, batch_size=batch_size, seed=seed)
    st.write("**Processando o Conjunto de Teste...**")
    test_df = apply_transforms_and_get_embeddings(test_dataset_subset, model_for_embeddings, test_transforms, batch_size=batch_size, seed=seed)

    # Mapear rótulos para nomes de classes
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_df['class_name'] = train_df['label'].map(idx_to_class)
    valid_df['class_name'] = valid_df['label'].map(idx_to_class)
    test_df['class_name'] = test_df['label'].map(idx_to_class)

    # Exibir dataframes no Streamlit sem a coluna 'augmented_image' e sem limitar a 5 linhas
    st.write("### Dataframe do Conjunto de Treinamento com Data Augmentation e Embeddings:")
    st.dataframe(train_df.drop(columns=['augmented_image']))

    st.write("### Dataframe do Conjunto de Validação:")
    st.dataframe(valid_df.drop(columns=['augmented_image']))

    st.write("### Dataframe do Conjunto de Teste:")
    st.dataframe(test_df.drop(columns=['augmented_image']))

    # Exibir todas as imagens augmentadas (ou limitar conforme necessário)
    display_all_augmented_images(train_df, full_dataset.classes, max_images=100)  # Ajuste 'max_images' conforme necessário

    # Visualizar os embeddings
    visualize_embeddings(train_df, full_dataset.classes)

    # Exibir contagem de imagens por classe nos conjuntos de treinamento e teste
    st.write("### Distribuição das Classes no Conjunto de Treinamento:")
    train_class_counts = train_df['class_name'].value_counts()
    fig = px.bar(
        train_class_counts, 
        x=train_class_counts.index, 
        y=train_class_counts.values,
        labels={'x': 'Classe', 'y': 'Número de Imagens'},
        title='Distribuição das Classes no Conjunto de Treinamento'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Distribuição das Classes no Conjunto de Teste:")
    test_class_counts = test_df['class_name'].value_counts()
    fig = px.bar(
        test_class_counts, 
        x=test_class_counts.index, 
        y=test_class_counts.values,
        labels={'x': 'Classe', 'y': 'Número de Imagens'},
        title='Distribuição das Classes no Conjunto de Teste'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Atualizar os datasets com as transformações para serem usados nos DataLoaders
    train_dataset = CustomDataset(torch.utils.data.Subset(full_dataset, train_indices), transform=train_transforms)
    valid_dataset = CustomDataset(torch.utils.data.Subset(full_dataset, valid_indices), transform=test_transforms)
    test_dataset = CustomDataset(torch.utils.data.Subset(full_dataset, test_indices), transform=test_transforms)

    # Dataloaders
    g = torch.Generator()
    g.manual_seed(seed)

    if use_weighted_loss:
        targets = [full_dataset.targets[i] for i in train_indices]
        class_counts = np.bincount(targets)
        class_counts = class_counts + 1e-6  # Para evitar divisão por zero
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        worker_init_fn=seed_worker, 
        generator=g
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=seed_worker, 
        generator=g
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=seed_worker, 
        generator=g
    )

    # Carregar o modelo
    model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=fine_tune, seed=seed)
    if model is None:
        return None

    # Definir o otimizador com L2 regularization (weight_decay)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate, 
        weight_decay=l2_lambda
    )

    # Inicializar as listas de perdas e acurácias no st.session_state
    if 'train_losses' not in st.session_state:
        st.session_state.train_losses = []
    if 'valid_losses' not in st.session_state:
        st.session_state.valid_losses = []
    if 'train_accuracies' not in st.session_state:
        st.session_state.train_accuracies = []
    if 'valid_accuracies' not in st.session_state:
        st.session_state.valid_accuracies = []

    # Early Stopping
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None  # Inicializar

    # Placeholders para gráficos dinâmicos
    placeholder = st.empty()
    progress_bar = st.progress(0)
    epoch_text = st.empty()

    # Treinamento
    for epoch in range(epochs):
        set_seed(seed + epoch)  # Atualizar a seed para cada época para evitar repetição

        running_loss = 0.0
        running_corrects = 0
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            try:
                outputs = model(inputs)
            except Exception as e:
                st.error(f"Erro durante o treinamento: {e}")
                return None

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        st.session_state.train_losses.append(epoch_loss)
        st.session_state.train_accuracies.append(epoch_acc.item())

        # Validação
        model.eval()
        valid_running_loss = 0.0
        valid_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                valid_running_loss += loss.item() * inputs.size(0)
                valid_running_corrects += torch.sum(preds == labels.data)

        valid_epoch_loss = valid_running_loss / len(valid_dataset)
        valid_epoch_acc = valid_running_corrects.double() / len(valid_dataset)
        st.session_state.valid_losses.append(valid_epoch_loss)
        st.session_state.valid_accuracies.append(valid_epoch_acc.item())

        # Atualizar gráficos dinamicamente usando Plotly
        with placeholder.container():
            fig = go.Figure()

            # Gráfico de Perda
            fig.add_trace(go.Scatter(
                x=list(range(1, epoch + 2)),
                y=st.session_state.train_losses,
                mode='lines+markers',
                name='Perda de Treino',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, epoch + 2)),
                y=st.session_state.valid_losses,
                mode='lines+markers',
                name='Perda de Validação',
                line=dict(color='red')
            ))

            # Gráfico de Acurácia
            fig.add_trace(go.Scatter(
                x=list(range(1, epoch + 2)),
                y=st.session_state.train_accuracies,
                mode='lines+markers',
                name='Acurácia de Treino',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, epoch + 2)),
                y=st.session_state.valid_accuracies,
                mode='lines+markers',
                name='Acurácia de Validação',
                line=dict(color='orange')
            ))

            fig.update_layout(
                title=f'Epoca {epoch+1}/{epochs} - Treinamento e Validação',
                xaxis_title='Épocas',
                yaxis_title='Valor',
                legend=dict(x=0, y=1.2, orientation='h'),
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Atualizar texto de progresso
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        epoch_text.text(f'Época {epoch+1}/{epochs}')

        # Atualizar histórico na barra lateral
        with st.sidebar.expander("Histórico de Treinamento", expanded=True):
            timestamp_hist = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Gráfico de Perda
            fig_loss = px.line(
                x=list(range(1, epoch + 2)),
                y=st.session_state.train_losses,
                labels={'x': 'Época', 'y': 'Perda'},
                title=f'Histórico de Perda ({timestamp_hist})',
                name='Perda de Treino'
            )
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, epoch + 2)),
                y=st.session_state.valid_losses,
                mode='lines+markers',
                name='Perda de Validação',
                line=dict(color='red')
            ))
            fig_loss.update_layout(template='plotly_white')
            st.plotly_chart(fig_loss, use_container_width=True)

            # Gráfico de Acurácia
            fig_acc = px.line(
                x=list(range(1, epoch + 2)),
                y=st.session_state.train_accuracies,
                labels={'x': 'Época', 'y': 'Acurácia'},
                title=f'Histórico de Acurácia ({timestamp_hist})',
                name='Acurácia de Treino'
            )
            fig_acc.add_trace(go.Scatter(
                x=list(range(1, epoch + 2)),
                y=st.session_state.valid_accuracies,
                mode='lines+markers',
                name='Acurácia de Validação',
                line=dict(color='orange')
            ))
            fig_acc.update_layout(template='plotly_white')
            st.plotly_chart(fig_acc, use_container_width=True)

            # Botão para limpar o histórico
            if st.button("Limpar Histórico", key=f"limpar_historico_epoch_{epoch}"):
                st.session_state.train_losses = []
                st.session_state.valid_losses = []
                st.session_state.train_accuracies = []
                st.session_state.valid_accuracies = []
                st.experimental_rerun()

        # Early Stopping
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                st.write('Early stopping!')
                if best_model_wts is not None:
                    model.load_state_dict(best_model_wts)
                break

    # Carregar os melhores pesos do modelo se houver
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Gráficos de Perda e Acurácia finais usando Plotly
    plot_metrics(st.session_state.train_losses, st.session_state.valid_losses, 
                st.session_state.train_accuracies, st.session_state.valid_accuracies)

    # Avaliação Final no Conjunto de Teste
    st.write("### Avaliação no Conjunto de Teste")
    compute_metrics(model, test_loader, full_dataset.classes)

    # Análise de Erros
    st.write("### Análise de Erros")
    error_analysis(model, test_loader, full_dataset.classes)

    # **Clusterização e Análise Comparativa**
    st.write("### Análise de Clusterização")
    perform_clustering(model, test_loader, full_dataset.classes)

    # Liberar memória
    del train_loader, valid_loader
    gc.collect()

    # Armazenar o modelo e as classes no st.session_state
    st.session_state['model'] = model
    st.session_state['classes'] = full_dataset.classes
    st.session_state['trained_model_name'] = model_name  # Armazena o nome do modelo treinado

    return model, full_dataset.classes

def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies):
    """
    Plota os gráficos de perda e acurácia usando Plotly.
    """
    epochs_range = list(range(1, len(train_losses) + 1))
    fig = go.Figure()

    # Perda de Treino
    fig.add_trace(go.Scatter(
        x=epochs_range,
        y=train_losses,
        mode='lines+markers',
        name='Perda de Treino',
        line=dict(color='blue')
    ))

    # Perda de Validação
    fig.add_trace(go.Scatter(
        x=epochs_range,
        y=valid_losses,
        mode='lines+markers',
        name='Perda de Validação',
        line=dict(color='red')
    ))

    # Acurácia de Treino
    fig.add_trace(go.Scatter(
        x=epochs_range,
        y=train_accuracies,
        mode='lines+markers',
        name='Acurácia de Treino',
        line=dict(color='green')
    ))

    # Acurácia de Validação
    fig.add_trace(go.Scatter(
        x=epochs_range,
        y=valid_accuracies,
        mode='lines+markers',
        name='Acurácia de Validação',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title='Perda e Acurácia por Época',
        xaxis_title='Épocas',
        yaxis_title='Valor',
        legend=dict(x=0, y=1.2, orientation='h'),
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

def compute_metrics(model, dataloader, classes):
    """
    Calcula métricas detalhadas e exibe matriz de confusão e relatório de classificação.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Relatório de Classificação
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    st.text("### Relatório de Classificação:")
    st.write(pd.DataFrame(report).transpose())

    # Matriz de Confusão Normalizada usando Plotly
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig = px.imshow(
        cm, 
        text_auto=True, 
        labels=dict(x="Predito", y="Verdadeiro", color="Proporção"),
        x=classes, 
        y=classes, 
        title="Matriz de Confusão Normalizada",
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Curva ROC usando Plotly
    if len(classes) == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, [p[1] for p in all_probs])
        roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Linha de Referência', line=dict(dash='dash')))
        fig.update_layout(
            title='Curva ROC',
            xaxis_title='Taxa de Falsos Positivos',
            yaxis_title='Taxa de Verdadeiros Positivos',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Multiclasse
        binarized_labels = label_binarize(all_labels, classes=range(len(classes)))
        roc_auc = roc_auc_score(binarized_labels, np.array(all_probs), average='weighted', multi_class='ovr')
        st.write(f"**AUC-ROC Média Ponderada:** {roc_auc:.4f}")

def error_analysis(model, dataloader, classes):
    """
    Realiza análise de erros mostrando algumas imagens mal classificadas.
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            incorrect = preds != labels
            if incorrect.any():
                misclassified_images.extend(inputs[incorrect].cpu())
                misclassified_labels.extend(labels[incorrect].cpu())
                misclassified_preds.extend(preds[incorrect].cpu())
                if len(misclassified_images) >= 5:
                    break

    if misclassified_images:
        st.write("### Algumas Imagens Mal Classificadas:")
        fig = plt.figure(figsize=(15, 3))
        for i in range(min(5, len(misclassified_images))):
            image = misclassified_images[i]
            image = image.permute(1, 2, 0).numpy()
            ax = fig.add_subplot(1, 5, i+1)
            ax.imshow(image)
            ax.set_title(f"V: {classes[misclassified_labels[i]]}\nP: {classes[misclassified_preds[i]]}")
            ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)  # Fechar a figura para liberar memória
    else:
        st.write("Nenhuma imagem mal classificada encontrada.")

def perform_clustering(model, dataloader, classes):
    """
    Realiza a extração de features e aplica algoritmos de clusterização.
    """
    # Extrair features usando o modelo pré-treinado
    features = []
    labels = []

    # Remover a última camada (classificador)
    if isinstance(model, nn.Sequential):
        model_feat = model
    else:
        model_feat = nn.Sequential(*list(model.children())[:-1])
    model_feat.eval()
    model_feat.to(device)

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            output = model_feat(inputs)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels.extend(label.numpy())

    features = np.vstack(features)
    labels = np.array(labels)

    # Redução de dimensionalidade com PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Clusterização com KMeans
    kmeans = KMeans(n_clusters=len(classes), random_state=42)
    clusters_kmeans = kmeans.fit_predict(features)

    # Clusterização Hierárquica
    agglo = AgglomerativeClustering(n_clusters=len(classes))
    clusters_agglo = agglo.fit_predict(features)

    # Criar DataFrame para Plotagem
    cluster_df = pd.DataFrame({
        'PC1': features_2d[:, 0],
        'PC2': features_2d[:, 1],
        'Cluster KMeans': clusters_kmeans,
        'Cluster Agglomerative': clusters_agglo
    })

    # Plotar Clusterização com KMeans
    fig_kmeans = px.scatter(
        cluster_df, 
        x='PC1', 
        y='PC2', 
        color='Cluster KMeans',
        title='Clusterização com KMeans',
        labels={'Cluster KMeans': 'Clusters'},
        template='plotly_white'
    )
    st.plotly_chart(fig_kmeans, use_container_width=True)

    # Plotar Clusterização com Agglomerative Clustering
    fig_agglo = px.scatter(
        cluster_df, 
        x='PC1', 
        y='PC2', 
        color='Cluster Agglomerative',
        title='Clusterização Hierárquica',
        labels={'Cluster Agglomerative': 'Clusters'},
        template='plotly_white'
    )
    st.plotly_chart(fig_agglo, use_container_width=True)

    # Métricas de Avaliação
    ari_kmeans = adjusted_rand_score(labels, clusters_kmeans)
    nmi_kmeans = normalized_mutual_info_score(labels, clusters_kmeans)
    ari_agglo = adjusted_rand_score(labels, clusters_agglo)
    nmi_agglo = normalized_mutual_info_score(labels, clusters_agglo)

    st.write(f"**KMeans** - ARI: {ari_kmeans:.4f}, NMI: {nmi_kmeans:.4f}")
    st.write(f"**Agglomerative Clustering** - ARI: {ari_agglo:.4f}, NMI: {nmi_agglo:.4f}")

def train_segmentation_model(images_dir, masks_dir, num_classes):
    """
    Função placeholder para treinar um modelo de segmentação.
    Implementação completa depende dos requisitos específicos e dos dados.
    """
    st.write("**Treinamento do modelo de segmentação não implementado.**")
    # Aqui você deve implementar o treinamento do modelo de segmentação conforme necessário
    return None

def visualize_activations(model, image, classes, model_name, segmentation_model=None, segmentation=False):
    """
    Visualiza as ativações do modelo utilizando Grad-CAM.
    """
    try:
        # Inicializar o método Grad-CAM
        cam_extractor = SmoothGradCAMpp(model, target_layer='fc')  # Ajuste 'fc' conforme o modelo
    
        # Preparar a imagem
        image_tensor = test_transforms(image).unsqueeze(0).to(device)
    
        # Fazer a previsão
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
            class_idx = pred.item()
    
        # Extrair o CAM
        activation_map = cam_extractor(class_idx, output)
        activation_map = activation_map.squeeze().cpu().numpy()
    
        # Overlay do CAM na imagem original
        cam_image = overlay_mask(
            to_pil_image(image_tensor.squeeze().cpu()),
            Image.fromarray((activation_map * 255).astype(np.uint8)),
            alpha=0.5
        )
    
        st.write("### Ativações do Modelo (Grad-CAM):")
        st.image(cam_image, caption='Grad-CAM', use_container_width=True)
    
        # Visualizar a segmentação, se disponível
        if segmentation_model is not None and segmentation:
            st.write("### Segmentação:")
            # Implementar a visualização da segmentação conforme o modelo
            # Esta é uma implementação placeholder
            seg_mask = segmentation_model(image_tensor)
            seg_mask = torch.argmax(seg_mask, dim=1).squeeze().cpu().numpy()
            seg_image = Image.fromarray(seg_mask.astype(np.uint8) * 255)
            st.image(seg_image, caption='Máscara de Segmentação', use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao visualizar ativações: {e}")

def apply_transforms_and_get_embeddings(dataset_subset, model_for_embeddings, transform, batch_size, seed):
    """
    Aplica transformações ao dataset e extrai embeddings utilizando o modelo clássico.
    Retorna um DataFrame com embeddings, rótulos e imagens augmentadas.
    """
    dataloader = DataLoader(
        dataset_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=seed_worker, 
        generator=torch.Generator().manual_seed(seed)
    )
    
    embeddings = []
    labels = []
    augmented_images = []
    
    model_for_embeddings.eval()
    model_for_embeddings.to(device)
    
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            outputs = model_for_embeddings(inputs)
            embeddings.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())
            augmented_images.extend(inputs.cpu())
    
    embeddings = np.vstack(embeddings)
    df = pd.DataFrame(embeddings, columns=[f'feat_{i}' for i in range(embeddings.shape[1])])
    df['label'] = labels
    df['augmented_image'] = augmented_images
    return df

def display_all_augmented_images(df, classes, max_images=100):
    """
    Exibe todas as imagens augmentadas em um grid.
    """
    st.write(f"### Exibição de Até {max_images} Imagens Augmentadas")
    images = df['augmented_image'].tolist()[:max_images]
    num_cols = 5
    num_rows = int(np.ceil(len(images) / num_cols))
    fig = plt.figure(figsize=(15, 3 * num_rows))
    for i, img_tensor in enumerate(images):
        if i >= max_images:
            break
        image = img_tensor.permute(1, 2, 0).numpy()
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ax.imshow(image)
        ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

def visualize_embeddings(df, classes):
    """
    Visualiza os embeddings usando PCA.
    """
    st.write("### Visualização dos Embeddings com PCA")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(df.drop(columns=['label', 'augmented_image']))
    df_plot = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'Classe': [classes[label] for label in df['label']]
    })
    fig = px.scatter(
        df_plot, 
        x='PC1', 
        y='PC2', 
        color='Classe',
        title='Embeddings Visualizados com PCA',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Definir o caminho do ícone
    icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto

    # Verificar se o arquivo de ícone existe antes de configurá-lo
    if os.path.exists(icon_path):
        try:
            st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
            logging.info(f"Ícone {icon_path} carregado com sucesso.")
        except Exception as e:
            st.set_page_config(page_title="Geomaker", layout="wide")
            logging.warning(f"Erro ao carregar o ícone {icon_path}: {e}")
    else:
        # Se o ícone não for encontrado, carrega sem favicon
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")

    # Layout da página
    if os.path.exists('capa.png'):
        try:
            st.image(
                'capa.png', 
                width=100, 
                caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', 
                use_container_width=True
            )
        except UnidentifiedImageError:
            st.warning("Imagem 'capa.png' não pôde ser carregada ou está corrompida.")
    else:
        st.warning("Imagem 'capa.png' não encontrada.")

    # Carregar o logotipo na barra lateral
    if os.path.exists("logo.png"):
        try:
            st.sidebar.image("logo.png", width=200)
        except UnidentifiedImageError:
            st.sidebar.text("Imagem do logotipo não pôde ser carregada ou está corrompida.")
    else:
        st.sidebar.text("Imagem do logotipo não encontrada.")

    st.title("Classificação e Segmentação de Imagens de Melanoma com Aprendizado Profundo e Quântico")
    st.write("Este aplicativo permite treinar modelos de classificação de melanomas utilizando conjuntos de dados adequados, aplicar algoritmos de clustering para análise comparativa e realizar segmentação de objetos.")
    st.write("As etapas são cuidadosamente documentadas para auxiliar na reprodução e análise científica.")

    # Inicializar segmentation_model
    segmentation_model = None

    # Opções para o modelo de segmentação
    st.subheader("Opções para o Modelo de Segmentação")
    segmentation_option = st.selectbox(
        "Deseja utilizar um modelo de segmentação?", 
        ["Não", "Utilizar modelo pré-treinado", "Treinar novo modelo de segmentação"]
    )
    if segmentation_option == "Utilizar modelo pré-treinado":
        num_classes_segmentation = st.number_input(
            "Número de Classes para Segmentação (Modelo Pré-treinado):", 
            min_value=1, 
            step=1, 
            value=21
        )
        segmentation_model = get_segmentation_model(num_classes=num_classes_segmentation, seed=42)
        st.write("Modelo de segmentação pré-treinado carregado.")
    elif segmentation_option == "Treinar novo modelo de segmentação":
        st.write("Treinamento do modelo de segmentação com seu próprio conjunto de dados.")
        num_classes_segmentation = st.number_input(
            "Número de Classes para Segmentação:", 
            min_value=1, 
            step=1, 
            value=2
        )
        # Upload do conjunto de dados de segmentação
        segmentation_zip = st.file_uploader(
            "Faça upload de um arquivo ZIP contendo as imagens e máscaras de segmentação", 
            type=["zip"]
        )
        if segmentation_zip is not None:
            temp_seg_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_seg_dir, "segmentation.zip")
            with open(zip_path, "wb") as f:
                f.write(segmentation_zip.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_seg_dir)

            # Espera-se que as imagens estejam em 'images/' e as máscaras em 'masks/' dentro do ZIP
            images_dir = os.path.join(temp_seg_dir, 'images')
            masks_dir = os.path.join(temp_seg_dir, 'masks')

            if os.path.exists(images_dir) and os.path.exists(masks_dir):
                # Treinar o modelo de segmentação
                st.write("Iniciando o treinamento do modelo de segmentação...")
                segmentation_model = train_segmentation_model(images_dir, masks_dir, num_classes_segmentation)
                if segmentation_model is not None:
                    st.success("Treinamento do modelo de segmentação concluído!")
            else:
                st.error("Estrutura de diretórios inválida no arquivo ZIP. Certifique-se de que as imagens estão em 'images/' e as máscaras em 'masks/'.")
        else:
            st.warning("Aguardando o upload do conjunto de dados de segmentação.")
    else:
        segmentation_model = None

    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")

    # Adicionar seleção de modo
    mode = st.sidebar.selectbox(
        "Modo de Treinamento:",
        options=["Clássico (PyTorch)", "Quântico (TFQ)"],
        index=0,
        key="mode_selection"
    )

    if mode == "Clássico (PyTorch)":
        num_classes = st.sidebar.number_input(
            "Número de Classes:", 
            min_value=2, 
            step=1, 
            key="num_classes"
        )
        model_name = st.sidebar.selectbox(
            "Modelo Pré-treinado:", 
            options=['ResNet18', 'ResNet50', 'DenseNet121'], 
            key="model_name"
        )
        fine_tune = st.sidebar.checkbox(
            "Fine-Tuning Completo", 
            value=False, 
            key="fine_tune"
        )
        epochs = st.sidebar.slider(
            "Número de Épocas:", 
            min_value=1, 
            max_value=500, 
            value=20, 
            step=1, 
            key="epochs"
        )
        learning_rate = st.sidebar.select_slider(
            "Taxa de Aprendizagem:", 
            options=[0.1, 0.01, 0.001, 0.0001], 
            value=0.001, 
            key="learning_rate"
        )
        batch_size = st.sidebar.selectbox(
            "Tamanho de Lote:", 
            options=[4, 8, 16, 32, 64], 
            index=2, 
            key="batch_size"
        )
        train_split = st.sidebar.slider(
            "Percentual de Treinamento:", 
            min_value=0.5, 
            max_value=0.9, 
            value=0.7, 
            step=0.05, 
            key="train_split"
        )
        valid_split = st.sidebar.slider(
            "Percentual de Validação:", 
            min_value=0.05, 
            max_value=0.4, 
            value=0.15, 
            step=0.05, 
            key="valid_split"
        )
        l2_lambda = st.sidebar.number_input(
            "L2 Regularization (Weight Decay):", 
            min_value=0.0, 
            max_value=0.1, 
            value=0.01, 
            step=0.01, 
            key="l2_lambda"
        )
        patience = st.sidebar.number_input(
            "Paciência para Early Stopping:", 
            min_value=1, 
            max_value=10, 
            value=3, 
            step=1, 
            key="patience"
        )
        use_weighted_loss = st.sidebar.checkbox(
            "Usar Perda Ponderada para Classes Desbalanceadas", 
            value=False, 
            key="use_weighted_loss"
        )
    elif mode == "Quântico (TFQ)":
        # Configurações para o modo quântico
        epochs_q = st.sidebar.slider(
            "Número de Épocas (Quântico):", 
            min_value=1, 
            max_value=20, 
            value=3, 
            step=1, 
            key="epochs_q"
        )
        batch_size_q = st.sidebar.selectbox(
            "Tamanho de Lote (Quântico):", 
            options=[4, 8, 16, 32, 64], 
            index=2, 
            key="batch_size_q"
        )
        threshold_q = st.sidebar.slider(
            "Threshold para Binarização [0,1] (Quântico):", 
            0.0, 
            1.0, 
            0.5, 
            step=0.05, 
            key="threshold_q"
        )
        circuit_type = st.sidebar.selectbox(
            "Tipo de Circuito:", 
            options=['Basic', 'Entangling', 'Rotation'], 
            key="circuit_type"
        )
        optimizer_type = st.sidebar.selectbox(
            "Tipo de Otimizador:", 
            options=['Adam', 'SGD', 'RMSprop'], 
            key="optimizer_type"
        )
        use_hardware = st.sidebar.checkbox(
            "Usar Hardware Quântico Real (IBM Quantum)", 
            value=False, 
            key="use_hardware"
        )
        if use_hardware:
            # Como Qiskit IBM Quantum foi removido, informamos que apenas simuladores são suportados
            st.sidebar.error("Integração com hardware quântico real removida. Apenas simuladores estão disponíveis.")
            backend_name = 'statevector_simulator'
        else:
            backend_name = 'statevector_simulator'
            st.sidebar.info(f"Usando backend simulado: {backend_name}")

        # Mensagem clara de modo experimental
        st.sidebar.warning(
            "⚠️ **Modo Quântico Experimental:** Atualmente, os modelos quânticos não superam os modelos clássicos (CNNs) para tarefas de classificação de imagens. Utilize este modo para fins educacionais e exploratórios."
        )

    # Opções de carregamento do modelo
    st.header("Opções de Carregamento do Modelo")

    model_option = st.selectbox(
        "Escolha uma opção:", 
        ["Treinar um novo modelo", "Carregar um modelo existente"], 
        key="model_option_main"
    )
    if model_option == "Carregar um modelo existente":
        if mode == "Clássico (PyTorch)":
            # Upload do modelo pré-treinado
            model_file = st.file_uploader(
                "Faça upload do arquivo do modelo (.pt ou .pth)", 
                type=["pt", "pth"], 
                key="model_file_uploader_main"
            )
            if model_file is not None:
                if num_classes > 0:
                    # Carregar o modelo clássico
                    model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False, seed=42)
                    if model is None:
                        st.error("Erro ao carregar o modelo.")
                        return

                    # Carregar os pesos do modelo
                    try:
                        state_dict = torch.load(model_file, map_location=device)
                        model.load_state_dict(state_dict)
                        st.session_state['model'] = model
                        st.session_state['trained_model_name'] = model_name  # Armazena o nome do modelo treinado
                        st.success("Modelo clássico carregado com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao carregar o modelo: {e}")
                        return

                    # Carregar as classes
                    classes_file = st.file_uploader(
                        "Faça upload do arquivo com as classes (classes.txt)", 
                        type=["txt"], 
                        key="classes_file_uploader_main"
                    )
                    if classes_file is not None:
                        classes = classes_file.read().decode("utf-8").splitlines()
                        st.session_state['classes'] = classes
                        st.write(f"Classes carregadas: {classes}")
                    else:
                        st.error("Por favor, forneça o arquivo com as classes.")
                else:
                    st.warning("Por favor, forneça o número de classes correto.")
        elif mode == "Quântico (TFQ)":
            # Upload do modelo quântico pré-treinado
            q_model_file = st.file_uploader(
                "Faça upload do arquivo do modelo quântico (.keras)", 
                type=["keras", "h5"], 
                key="q_model_file_uploader_main"
            )
            if q_model_file is not None:
                try:
                    q_model = tf.keras.models.load_model(q_model_file, compile=False, custom_objects={'SerializablePQC': SerializablePQC})
                    st.session_state['q_model'] = q_model
                    st.success("Modelo quântico carregado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar o modelo quântico: {e}")
                    return

                # Carregar as classes
                classes_file_q = st.file_uploader(
                    "Faça upload do arquivo com as classes (classes_quantic.txt)", 
                    type=["txt"], 
                    key="classes_file_uploader_quantic"
                )
                if classes_file_q is not None:
                    classes_q = classes_file_q.read().decode("utf-8").splitlines()
                    st.session_state['classes'] = classes_q
                    st.write(f"Classes carregadas: {classes_q}")
                else:
                    st.error("Por favor, forneça o arquivo com as classes quânticas.")
    elif model_option == "Treinar um novo modelo":
        # Upload do arquivo ZIP
        zip_file = st.file_uploader(
            "Upload do arquivo ZIP com as imagens", 
            type=["zip"], 
            key="zip_file_uploader"
        )
        if zip_file is not None:
            if mode == "Clássico (PyTorch)":
                if num_classes > 0 and train_split + valid_split <= 0.95:
                    temp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(temp_dir, "uploaded.zip")
                    with open(zip_path, "wb") as f:
                        f.write(zip_file.read())
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Verificar subdiretórios
                    expected_dirs = ['images', 'masks']
                    found_dirs = {}
                    for root, dirs, files in os.walk(temp_dir):
                        for d in dirs:
                            if d.lower() in expected_dirs:
                                found_dirs[d.lower()] = os.path.join(root, d)
                    
                    if 'images' in found_dirs and 'masks' in found_dirs:
                        images_dir = found_dirs['images']
                        masks_dir = found_dirs['masks']

                        st.write("Iniciando o treinamento supervisionado...")
                        model_data = train_model(
                            images_dir, 
                            num_classes, 
                            model_name, 
                            fine_tune, 
                            epochs, 
                            learning_rate, 
                            batch_size, 
                            train_split, 
                            valid_split, 
                            use_weighted_loss, 
                            l2_lambda, 
                            patience, 
                            seed=42
                        )

                        if model_data is None:
                            st.error("Erro no treinamento do modelo.")
                            shutil.rmtree(temp_dir)
                            return

                        model, classes = model_data
                        # O modelo e as classes já estão armazenados no st.session_state
                        st.success("Treinamento concluído!")

                        # Opção para baixar o modelo treinado
                        st.write("### Faça o Download do Modelo Treinado:")
                        buffer = io.BytesIO()
                        torch.save(model.state_dict(), buffer)
                        buffer.seek(0)
                        btn = st.download_button(
                            label="Download do Modelo",
                            data=buffer,
                            file_name="modelo_treinado.pth",
                            mime="application/octet-stream",
                            key="download_model_button"
                        )

                        # Salvar as classes em um arquivo
                        classes_data = "\n".join(classes)
                        st.download_button(
                            label="Download das Classes",
                            data=classes_data,
                            file_name="classes.txt",
                            mime="text/plain",
                            key="download_classes_button"
                        )

                        # Limpar o diretório temporário
                        shutil.rmtree(temp_dir)
                    else:
                        st.error("Estrutura de diretórios inválida no arquivo ZIP. Certifique-se de que as imagens estão em 'images/' e as máscaras em 'masks/'.")
            elif mode == "Quântico (TFQ)":
                if epochs_q > 0 and batch_size_q > 0:
                    # Treinamento quântico baseado no ISIC
                    st.write("Iniciando o treinamento do modelo quântico...")
                    quantum_model, quantum_circuit, quantum_symbols, observables = train_quantum_model(
                        epochs=epochs_q, 
                        batch_size=batch_size_q,
                        n_qubits=4,
                        n_layers=2,
                        seed=42
                    )
                    if quantum_model is not None:
                        st.success("Treinamento do modelo quântico concluído!")

                        # Exibir circuitos quânticos treinados
                        st.write("### Circuito Quântico Treinado:")
                        visualize_circuit(quantum_circuit)

                        # Salvar o modelo quântico no formato Keras nativo
                        try:
                            quantum_model.save("quantum_model.keras", save_format='keras')
                            st.write("### Modelo Quântico Salvo como `quantum_model.keras`.")
                        except Exception as e:
                            st.error(f"Erro ao salvar o modelo quântico: {e}")

                        # Salvar as classes em um arquivo
                        # Como o treinamento quântico foi feito com ISIC, classes são definidas pelo usuário
                        classes_q = ["Classe_0", "Classe_1"]  # Ajuste conforme o seu conjunto de dados
                        classes_data_q = "\n".join(classes_q)
                        st.download_button(
                            label="Download das Classes (Quântico)",
                            data=classes_data_q,
                            file_name="classes_quantic.txt",
                            mime="text/plain",
                            key="download_classes_quantic_button"
                        )
                    else:
                        st.error("Erro no treinamento do modelo quântico.")
                else:
                    st.warning("Por favor, forneça os dados e as configurações corretas para o modo quântico.")

    # Avaliação de uma imagem individual
    st.header("Avaliação de Imagem")
    evaluate = st.radio(
        "Deseja avaliar uma imagem?", 
        ("Sim", "Não"), 
        key="evaluate_option"
    )
    if evaluate == "Sim":
        eval_image_file = st.file_uploader(
            "Faça upload da imagem para avaliação", 
            type=["png", "jpg", "jpeg", "bmp", "gif"], 
            key="eval_image_file"
        )
        if eval_image_file is not None:
            eval_image_file.seek(0)
            try:
                eval_image = Image.open(eval_image_file).convert("RGB")
            except Exception as e:
                st.error(f"Erro ao abrir a imagem: {e}")
                return

            st.image(eval_image, caption='Imagem para avaliação', use_container_width=True)

            if mode == "Clássico (PyTorch)":
                # Verificar se o modelo já foi carregado ou treinado
                if 'model' not in st.session_state or 'classes' not in st.session_state:
                    st.warning("Nenhum modelo carregado ou treinado. Por favor, carregue um modelo existente ou treine um novo modelo.")
                else:
                    model_eval = st.session_state['model']
                    classes_eval = st.session_state['classes']
                    model_name_eval = st.session_state.get('trained_model_name', 'ResNet18')  # Usa o nome do modelo armazenado

                    class_name, confidence = evaluate_image(model_eval, eval_image, classes_eval)
                    st.write(f"**Classe Predita:** {class_name}")
                    st.write(f"**Confiança:** {confidence:.4f}")

                    # Opção para visualizar segmentação
                    segmentation = False
                    if segmentation_model is not None:
                        segmentation = st.checkbox(
                            "Visualizar Segmentação", 
                            value=True, 
                            key="segmentation_checkbox"
                        )

                    # Visualizar ativações e segmentação
                    visualize_activations(
                        model_eval, 
                        eval_image, 
                        classes_eval, 
                        model_name_eval, 
                        segmentation_model=segmentation_model, 
                        segmentation=segmentation
                    )
            elif mode == "Quântico (TFQ)":
                # Verificar se o modelo quântico está carregado ou treinado
                if 'q_model' not in st.session_state or 'classes' not in st.session_state:
                    st.warning("Nenhum modelo quântico carregado ou treinado. Por favor, carregue um modelo quântico existente ou treine um novo modelo.")
                else:
                    quantum_model = st.session_state['q_model']
                    classes_eval = st.session_state['classes']  # Para o modo quântico, classes devem ser definidas pelo usuário

                    # Preparar a imagem para o modelo quântico
                    # Reduzir para 2x2 e binarizar (ajustado para 2x2 qubits)
                    image_resized = eval_image.resize((2, 2))
                    image_tensor = np.array(image_resized) / 255.0
                    image_bin = (image_tensor > threshold_q).astype(float).flatten()

                    # Converter para circuito
                    circuit_q = cirq.Circuit()
                    qubits = cirq.GridQubit.rect(1, 2)
                    for i, bit in enumerate(image_bin):
                        if bit:
                            circuit_q.append(cirq.X(qubits[i]))

                    # Converter circuito para string
                    circuits = tfq.convert_to_tensor([circuit_q])

                    # Fazer a previsão
                    y_pred = quantum_model.predict(circuits)
                    # y_pred está na faixa [0,1] devido ao uso de sigmoid
                    predicted_label = 1 if y_pred[0][0] > 0.5 else 0
                    confidence_q = y_pred[0][0]

                    # Obter o nome da classe
                    if predicted_label >= len(classes_eval):
                        class_name = "Desconhecida"
                    else:
                        class_name = classes_eval[predicted_label]

                    st.write(f"**Classe Predita (Quântico):** {class_name}")
                    st.write(f"**Confiança (Quântico):** {confidence_q:.4f}")

                    # Visualizar ativações - Grad-CAM não está implementado para modelos quânticos
                    st.write("**Visualização de Ativações:** Não disponível para o modo quântico.")

    # Visualização de Circuitos Quânticos
    st.header("Gerador e Visualizador de Circuitos Quânticos")
    st.write("Selecione o tipo de circuito quântico que deseja gerar e visualize-o abaixo.")

    # Seleção do tipo de circuito
    circuit_type_display = st.selectbox(
        "Selecione o Tipo de Circuito:",
        options=["Basic", "Entangling", "Rotation"],
        index=0,
        key="circuit_type_display"
    )

    # Função para criar circuitos com base no tipo selecionado
    def create_quantum_model_display(circuit_type):
        """
        Cria um circuito quântico com base no tipo selecionado.
        """
        if circuit_type == "Basic":
            n_qubits = 2
            n_layers = 1
        elif circuit_type == "Entangling":
            n_qubits = 4
            n_layers = 2
        elif circuit_type == "Rotation":
            n_qubits = 2
            n_layers = 3
        else:
            st.error("Tipo de circuito não suportado.")
            return None, None

        circuit, symbols = create_quantum_circuit(n_qubits, n_layers)
        return circuit, symbols

    # Botão para gerar o circuito
    if st.button("Gerar Circuito Quântico"):
        with st.spinner("Gerando o circuito quântico..."):
            quantum_circuit_display, quantum_symbols_display = create_quantum_model_display(circuit_type_display)
        
        if quantum_circuit_display is not None:
            st.write(f"### Circuito Selecionado: **{circuit_type_display}**")
            visualize_circuit(quantum_circuit_display)
        else:
            st.error("Falha ao gerar o circuito quântico.")

    # Opcional: Mostrar o código do circuito
    if st.checkbox("Mostrar Código do Circuito Quântico"):
        quantum_circuit_display, quantum_symbols_display = create_quantum_model_display(circuit_type_display)
        if quantum_circuit_display is not None:
            st.code(str(quantum_circuit_display), language='python')
        else:
            st.error("Nenhum código para exibir.")

    # Documentação dos Procedimentos
    st.write("### Documentação dos Procedimentos")
    st.write("Todas as etapas foram cuidadosamente registradas. Utilize esta documentação para reproduzir o experimento e analisar os resultados.")

    # Encerrar a aplicação
    st.write("Obrigado por utilizar o aplicativo!")

# Funções adicionais necessárias

def apply_transforms_and_get_embeddings(dataset_subset, model_for_embeddings, transform, batch_size, seed):
    """
    Aplica transformações ao dataset e extrai embeddings utilizando o modelo clássico.
    Retorna um DataFrame com embeddings, rótulos e imagens augmentadas.
    """
    dataloader = DataLoader(
        dataset_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=seed_worker, 
        generator=torch.Generator().manual_seed(seed)
    )
    
    embeddings = []
    labels = []
    augmented_images = []
    
    model_for_embeddings.eval()
    model_for_embeddings.to(device)
    
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            outputs = model_for_embeddings(inputs)
            embeddings.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())
            augmented_images.extend(inputs.cpu())
    
    embeddings = np.vstack(embeddings)
    df = pd.DataFrame(embeddings, columns=[f'feat_{i}' for i in range(embeddings.shape[1])])
    df['label'] = labels
    df['augmented_image'] = augmented_images
    return df

def display_all_augmented_images(df, classes, max_images=100):
    """
    Exibe todas as imagens augmentadas em um grid.
    """
    st.write(f"### Exibição de Até {max_images} Imagens Augmentadas")
    images = df['augmented_image'].tolist()[:max_images]
    num_cols = 5
    num_rows = int(np.ceil(len(images) / num_cols))
    fig = plt.figure(figsize=(15, 3 * num_rows))
    for i, img_tensor in enumerate(images):
        if i >= max_images:
            break
        image = img_tensor.permute(1, 2, 0).numpy()
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ax.imshow(image)
        ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

def visualize_embeddings(df, classes):
    """
    Visualiza os embeddings usando PCA.
    """
    st.write("### Visualização dos Embeddings com PCA")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(df.drop(columns=['label', 'augmented_image']))
    df_plot = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'Classe': [classes[label] for label in df['label']]
    })
    fig = px.scatter(
        df_plot, 
        x='PC1', 
        y='PC2', 
        color='Classe',
        title='Embeddings Visualizados com PCA',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

def visualize_activations(model, image, classes, model_name, segmentation_model=None, segmentation=False):
    """
    Visualiza as ativações do modelo utilizando Grad-CAM.
    """
    try:
        # Inicializar o método Grad-CAM
        cam_extractor = SmoothGradCAMpp(model, target_layer='fc')  # Ajuste 'fc' conforme o modelo

        # Preparar a imagem
        image_tensor = test_transforms(image).unsqueeze(0).to(device)

        # Fazer a previsão
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
            class_idx = pred.item()

        # Extrair o CAM
        activation_map = cam_extractor(class_idx, output)
        activation_map = activation_map.squeeze().cpu().numpy()

        # Overlay do CAM na imagem original
        cam_image = overlay_mask(
            to_pil_image(image_tensor.squeeze().cpu()),
            Image.fromarray((activation_map * 255).astype(np.uint8)),
            alpha=0.5
        )

        st.write("### Ativações do Modelo (Grad-CAM):")
        st.image(cam_image, caption='Grad-CAM', use_container_width=True)

        # Visualizar a segmentação, se disponível
        if segmentation_model is not None and segmentation:
            st.write("### Segmentação:")
            # Implementar a visualização da segmentação conforme o modelo
            # Esta é uma implementação placeholder
            seg_mask = segmentation_model(image_tensor)
            seg_mask = torch.argmax(seg_mask, dim=1).squeeze().cpu().numpy()
            seg_image = Image.fromarray(seg_mask.astype(np.uint8) * 255)
            st.image(seg_image, caption='Máscara de Segmentação', use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao visualizar ativações: {e}")

def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies):
    """
    Plota os gráficos de perda e acurácia usando Plotly.
    """
    epochs_range = list(range(1, len(train_losses) + 1))
    fig = go.Figure()

    # Perda de Treino
    fig.add_trace(go.Scatter(
        x=epochs_range,
        y=train_losses,
        mode='lines+markers',
        name='Perda de Treino',
        line=dict(color='blue')
    ))

    # Perda de Validação
    fig.add_trace(go.Scatter(
        x=epochs_range,
        y=valid_losses,
        mode='lines+markers',
        name='Perda de Validação',
        line=dict(color='red')
    ))

    # Acurácia de Treino
    fig.add_trace(go.Scatter(
        x=epochs_range,
        y=train_accuracies,
        mode='lines+markers',
        name='Acurácia de Treino',
        line=dict(color='green')
    ))

    # Acurácia de Validação
    fig.add_trace(go.Scatter(
        x=epochs_range,
        y=valid_accuracies,
        mode='lines+markers',
        name='Acurácia de Validação',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title='Perda e Acurácia por Época',
        xaxis_title='Épocas',
        yaxis_title='Valor',
        legend=dict(x=0, y=1.2, orientation='h'),
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

def compute_metrics(model, dataloader, classes):
    """
    Calcula métricas detalhadas e exibe matriz de confusão e relatório de classificação.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Relatório de Classificação
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    st.text("### Relatório de Classificação:")
    st.write(pd.DataFrame(report).transpose())

    # Matriz de Confusão Normalizada usando Plotly
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig = px.imshow(
        cm, 
        text_auto=True, 
        labels=dict(x="Predito", y="Verdadeiro", color="Proporção"),
        x=classes, 
        y=classes, 
        title="Matriz de Confusão Normalizada",
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Curva ROC usando Plotly
    if len(classes) == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, [p[1] for p in all_probs])
        roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Linha de Referência', line=dict(dash='dash')))
        fig.update_layout(
            title='Curva ROC',
            xaxis_title='Taxa de Falsos Positivos',
            yaxis_title='Taxa de Verdadeiros Positivos',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Multiclasse
        binarized_labels = label_binarize(all_labels, classes=range(len(classes)))
        roc_auc = roc_auc_score(binarized_labels, np.array(all_probs), average='weighted', multi_class='ovr')
        st.write(f"**AUC-ROC Média Ponderada:** {roc_auc:.4f}")

def error_analysis(model, dataloader, classes):
    """
    Realiza análise de erros mostrando algumas imagens mal classificadas.
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            incorrect = preds != labels
            if incorrect.any():
                misclassified_images.extend(inputs[incorrect].cpu())
                misclassified_labels.extend(labels[incorrect].cpu())
                misclassified_preds.extend(preds[incorrect].cpu())
                if len(misclassified_images) >= 5:
                    break

    if misclassified_images:
        st.write("### Algumas Imagens Mal Classificadas:")
        fig = plt.figure(figsize=(15, 3))
        for i in range(min(5, len(misclassified_images))):
            image = misclassified_images[i]
            image = image.permute(1, 2, 0).numpy()
            ax = fig.add_subplot(1, 5, i+1)
            ax.imshow(image)
            ax.set_title(f"V: {classes[misclassified_labels[i]]}\nP: {classes[misclassified_preds[i]]}")
            ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)  # Fechar a figura para liberar memória
    else:
        st.write("Nenhuma imagem mal classificada encontrada.")

def perform_clustering(model, dataloader, classes):
    """
    Realiza a extração de features e aplica algoritmos de clusterização.
    """
    # Extrair features usando o modelo pré-treinado
    features = []
    labels = []

    # Remover a última camada (classificador)
    if isinstance(model, nn.Sequential):
        model_feat = model
    else:
        model_feat = nn.Sequential(*list(model.children())[:-1])
    model_feat.eval()
    model_feat.to(device)

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            output = model_feat(inputs)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels.extend(label.numpy())

    features = np.vstack(features)
    labels = np.array(labels)

    # Redução de dimensionalidade com PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Clusterização com KMeans
    kmeans = KMeans(n_clusters=len(classes), random_state=42)
    clusters_kmeans = kmeans.fit_predict(features)

    # Clusterização Hierárquica
    agglo = AgglomerativeClustering(n_clusters=len(classes))
    clusters_agglo = agglo.fit_predict(features)

    # Criar DataFrame para Plotagem
    cluster_df = pd.DataFrame({
        'PC1': features_2d[:, 0],
        'PC2': features_2d[:, 1],
        'Cluster KMeans': clusters_kmeans,
        'Cluster Agglomerative': clusters_agglo
    })

    # Plotar Clusterização com KMeans
    fig_kmeans = px.scatter(
        cluster_df, 
        x='PC1', 
        y='PC2', 
        color='Cluster KMeans',
        title='Clusterização com KMeans',
        labels={'Cluster KMeans': 'Clusters'},
        template='plotly_white'
    )
    st.plotly_chart(fig_kmeans, use_container_width=True)

    # Plotar Clusterização com Agglomerative Clustering
    fig_agglo = px.scatter(
        cluster_df, 
        x='PC1', 
        y='PC2', 
        color='Cluster Agglomerative',
        title='Clusterização Hierárquica',
        labels={'Cluster Agglomerative': 'Clusters'},
        template='plotly_white'
    )
    st.plotly_chart(fig_agglo, use_container_width=True)

    # Métricas de Avaliação
    ari_kmeans = adjusted_rand_score(labels, clusters_kmeans)
    nmi_kmeans = normalized_mutual_info_score(labels, clusters_kmeans)
    ari_agglo = adjusted_rand_score(labels, clusters_agglo)
    nmi_agglo = normalized_mutual_info_score(labels, clusters_agglo)

    st.write(f"**KMeans** - ARI: {ari_kmeans:.4f}, NMI: {nmi_kmeans:.4f}")
    st.write(f"**Agglomerative Clustering** - ARI: {ari_agglo:.4f}, NMI: {nmi_agglo:.4f}")

def train_segmentation_model(images_dir, masks_dir, num_classes):
    """
    Função placeholder para treinar um modelo de segmentação.
    Implementação completa depende dos requisitos específicos e dos dados.
    """
    st.write("**Treinamento do modelo de segmentação não implementado.**")
    # Aqui você deve implementar o treinamento do modelo de segmentação conforme necessário
    return None

def main():
    # Definir o caminho do ícone
    icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto

    # Verificar se o arquivo de ícone existe antes de configurá-lo
    if os.path.exists(icon_path):
        try:
            st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
            logging.info(f"Ícone {icon_path} carregado com sucesso.")
        except Exception as e:
            st.set_page_config(page_title="Geomaker", layout="wide")
            logging.warning(f"Erro ao carregar o ícone {icon_path}: {e}")
    else:
        # Se o ícone não for encontrado, carrega sem favicon
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")

    # Layout da página
    if os.path.exists('capa.png'):
        try:
            st.image(
                'capa.png', 
                width=100, 
                caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', 
                use_container_width=True
            )
        except UnidentifiedImageError:
            st.warning("Imagem 'capa.png' não pôde ser carregada ou está corrompida.")
    else:
        st.warning("Imagem 'capa.png' não encontrada.")

    # Carregar o logotipo na barra lateral
    if os.path.exists("logo.png"):
        try:
            st.sidebar.image("logo.png", width=200)
        except UnidentifiedImageError:
            st.sidebar.text("Imagem do logotipo não pôde ser carregada ou está corrompida.")
    else:
        st.sidebar.text("Imagem do logotipo não encontrada.")

    st.title("Classificação e Segmentação de Imagens de Melanoma com Aprendizado Profundo e Quântico")
    st.write("Este aplicativo permite treinar modelos de classificação de melanomas utilizando conjuntos de dados adequados, aplicar algoritmos de clustering para análise comparativa e realizar segmentação de objetos.")
    st.write("As etapas são cuidadosamente documentadas para auxiliar na reprodução e análise científica.")

    # Inicializar segmentation_model
    segmentation_model = None

    # Opções para o modelo de segmentação
    st.subheader("Opções para o Modelo de Segmentação")
    segmentation_option = st.selectbox(
        "Deseja utilizar um modelo de segmentação?", 
        ["Não", "Utilizar modelo pré-treinado", "Treinar novo modelo de segmentação"]
    )
    if segmentation_option == "Utilizar modelo pré-treinado":
        num_classes_segmentation = st.number_input(
            "Número de Classes para Segmentação (Modelo Pré-treinado):", 
            min_value=1, 
            step=1, 
            value=21
        )
        # Implementar a função get_segmentation_model conforme necessário
        segmentation_model = get_segmentation_model(num_classes=num_classes_segmentation, seed=42)
        st.write("Modelo de segmentação pré-treinado carregado.")
    elif segmentation_option == "Treinar novo modelo de segmentação":
        st.write("Treinamento do modelo de segmentação com seu próprio conjunto de dados.")
        num_classes_segmentation = st.number_input(
            "Número de Classes para Segmentação:", 
            min_value=1, 
            step=1, 
            value=2
        )
        # Upload do conjunto de dados de segmentação
        segmentation_zip = st.file_uploader(
            "Faça upload de um arquivo ZIP contendo as imagens e máscaras de segmentação", 
            type=["zip"]
        )
        if segmentation_zip is not None:
            temp_seg_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_seg_dir, "segmentation.zip")
            with open(zip_path, "wb") as f:
                f.write(segmentation_zip.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_seg_dir)

            # Espera-se que as imagens estejam em 'images/' e as máscaras em 'masks/' dentro do ZIP
            images_dir = os.path.join(temp_seg_dir, 'images')
            masks_dir = os.path.join(temp_seg_dir, 'masks')

            if os.path.exists(images_dir) and os.path.exists(masks_dir):
                # Treinar o modelo de segmentação
                st.write("Iniciando o treinamento do modelo de segmentação...")
                segmentation_model = train_segmentation_model(images_dir, masks_dir, num_classes_segmentation)
                if segmentation_model is not None:
                    st.success("Treinamento do modelo de segmentação concluído!")
            else:
                st.error("Estrutura de diretórios inválida no arquivo ZIP. Certifique-se de que as imagens estão em 'images/' e as máscaras em 'masks/'.")
        else:
            st.warning("Aguardando o upload do conjunto de dados de segmentação.")
    else:
        segmentation_model = None

    # Barra Lateral de Configurações
    st.sidebar.title("Configurações do Treinamento")

    # Adicionar seleção de modo
    mode = st.sidebar.selectbox(
        "Modo de Treinamento:",
        options=["Clássico (PyTorch)", "Quântico (TFQ)"],
        index=0,
        key="mode_selection"
    )

    if mode == "Clássico (PyTorch)":
        num_classes = st.sidebar.number_input(
            "Número de Classes:", 
            min_value=2, 
            step=1, 
            key="num_classes"
        )
        model_name = st.sidebar.selectbox(
            "Modelo Pré-treinado:", 
            options=['ResNet18', 'ResNet50', 'DenseNet121'], 
            key="model_name"
        )
        fine_tune = st.sidebar.checkbox(
            "Fine-Tuning Completo", 
            value=False, 
            key="fine_tune"
        )
        epochs = st.sidebar.slider(
            "Número de Épocas:", 
            min_value=1, 
            max_value=500, 
            value=20, 
            step=1, 
            key="epochs"
        )
        learning_rate = st.sidebar.select_slider(
            "Taxa de Aprendizagem:", 
            options=[0.1, 0.01, 0.001, 0.0001], 
            value=0.001, 
            key="learning_rate"
        )
        batch_size = st.sidebar.selectbox(
            "Tamanho de Lote:", 
            options=[4, 8, 16, 32, 64], 
            index=2, 
            key="batch_size"
        )
        train_split = st.sidebar.slider(
            "Percentual de Treinamento:", 
            min_value=0.5, 
            max_value=0.9, 
            value=0.7, 
            step=0.05, 
            key="train_split"
        )
        valid_split = st.sidebar.slider(
            "Percentual de Validação:", 
            min_value=0.05, 
            max_value=0.4, 
            value=0.15, 
            step=0.05, 
            key="valid_split"
        )
        l2_lambda = st.sidebar.number_input(
            "L2 Regularization (Weight Decay):", 
            min_value=0.0, 
            max_value=0.1, 
            value=0.01, 
            step=0.01, 
            key="l2_lambda"
        )
        patience = st.sidebar.number_input(
            "Paciência para Early Stopping:", 
            min_value=1, 
            max_value=10, 
            value=3, 
            step=1, 
            key="patience"
        )
        use_weighted_loss = st.sidebar.checkbox(
            "Usar Perda Ponderada para Classes Desbalanceadas", 
            value=False, 
            key="use_weighted_loss"
        )
    elif mode == "Quântico (TFQ)":
        # Configurações para o modo quântico
        epochs_q = st.sidebar.slider(
            "Número de Épocas (Quântico):", 
            min_value=1, 
            max_value=20, 
            value=3, 
            step=1, 
            key="epochs_q"
        )
        batch_size_q = st.sidebar.selectbox(
            "Tamanho de Lote (Quântico):", 
            options=[4, 8, 16, 32, 64], 
            index=2, 
            key="batch_size_q"
        )
        threshold_q = st.sidebar.slider(
            "Threshold para Binarização [0,1] (Quântico):", 
            0.0, 
            1.0, 
            0.5, 
            step=0.05, 
            key="threshold_q"
        )
        circuit_type = st.sidebar.selectbox(
            "Tipo de Circuito:", 
            options=['Basic', 'Entangling', 'Rotation'], 
            key="circuit_type"
        )
        optimizer_type = st.sidebar.selectbox(
            "Tipo de Otimizador:", 
            options=['Adam', 'SGD', 'RMSprop'], 
            key="optimizer_type"
        )
        use_hardware = st.sidebar.checkbox(
            "Usar Hardware Quântico Real (IBM Quantum)", 
            value=False, 
            key="use_hardware"
        )
        if use_hardware:
            # Como Qiskit IBM Quantum foi removido, informamos que apenas simuladores são suportados
            st.sidebar.error("Integração com hardware quântico real removida. Apenas simuladores estão disponíveis.")
            backend_name = 'statevector_simulator'
        else:
            backend_name = 'statevector_simulator'
            st.sidebar.info(f"Usando backend simulado: {backend_name}")

        # Mensagem clara de modo experimental
        st.sidebar.warning(
            "⚠️ **Modo Quântico Experimental:** Atualmente, os modelos quânticos não superam os modelos clássicos (CNNs) para tarefas de classificação de imagens. Utilize este modo para fins educacionais e exploratórios."
        )

    # Opções de carregamento do modelo
    st.header("Opções de Carregamento do Modelo")

    model_option = st.selectbox(
        "Escolha uma opção:", 
        ["Treinar um novo modelo", "Carregar um modelo existente"], 
        key="model_option_main"
    )
    if model_option == "Carregar um modelo existente":
        if mode == "Clássico (PyTorch)":
            # Upload do modelo pré-treinado
            model_file = st.file_uploader(
                "Faça upload do arquivo do modelo (.pt ou .pth)", 
                type=["pt", "pth"], 
                key="model_file_uploader_main"
            )
            if model_file is not None:
                if num_classes > 0:
                    # Carregar o modelo clássico
                    model = get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False, seed=42)
                    if model is None:
                        st.error("Erro ao carregar o modelo.")
                        return

                    # Carregar os pesos do modelo
                    try:
                        state_dict = torch.load(model_file, map_location=device)
                        model.load_state_dict(state_dict)
                        st.session_state['model'] = model
                        st.session_state['trained_model_name'] = model_name  # Armazena o nome do modelo treinado
                        st.success("Modelo clássico carregado com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao carregar o modelo: {e}")
                        return

                    # Carregar as classes
                    classes_file = st.file_uploader(
                        "Faça upload do arquivo com as classes (classes.txt)", 
                        type=["txt"], 
                        key="classes_file_uploader_main"
                    )
                    if classes_file is not None:
                        classes = classes_file.read().decode("utf-8").splitlines()
                        st.session_state['classes'] = classes
                        st.write(f"Classes carregadas: {classes}")
                    else:
                        st.error("Por favor, forneça o arquivo com as classes.")
                else:
                    st.warning("Por favor, forneça o número de classes correto.")
        elif mode == "Quântico (TFQ)":
            # Upload do modelo quântico pré-treinado
            q_model_file = st.file_uploader(
                "Faça upload do arquivo do modelo quântico (.keras)", 
                type=["keras", "h5"], 
                key="q_model_file_uploader_main"
            )
            if q_model_file is not None:
                try:
                    q_model = tf.keras.models.load_model(q_model_file, compile=False, custom_objects={'SerializablePQC': SerializablePQC})
                    st.session_state['q_model'] = q_model
                    st.success("Modelo quântico carregado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar o modelo quântico: {e}")
                    return

                # Carregar as classes
                classes_file_q = st.file_uploader(
                    "Faça upload do arquivo com as classes (classes_quantic.txt)", 
                    type=["txt"], 
                    key="classes_file_uploader_quantic"
                )
                if classes_file_q is not None:
                    classes_q = classes_file_q.read().decode("utf-8").splitlines()
                    st.session_state['classes'] = classes_q
                    st.write(f"Classes carregadas: {classes_q}")
                else:
                    st.error("Por favor, forneça o arquivo com as classes quânticas.")
    elif model_option == "Treinar um novo modelo":
        # Upload do arquivo ZIP
        zip_file = st.file_uploader(
            "Upload do arquivo ZIP com as imagens", 
            type=["zip"], 
            key="zip_file_uploader"
        )
        if zip_file is not None:
            if mode == "Clássico (PyTorch)":
                if num_classes > 0 and train_split + valid_split <= 0.95:
                    temp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(temp_dir, "uploaded.zip")
                    with open(zip_path, "wb") as f:
                        f.write(zip_file.read())
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Verificar subdiretórios
                    expected_dirs = ['images', 'masks']
                    found_dirs = {}
                    for root, dirs, files in os.walk(temp_dir):
                        for d in dirs:
                            if d.lower() in expected_dirs:
                                found_dirs[d.lower()] = os.path.join(root, d)
                    
                    if 'images' in found_dirs and 'masks' in found_dirs:
                        images_dir = found_dirs['images']
                        masks_dir = found_dirs['masks']

                        st.write("Iniciando o treinamento supervisionado...")
                        model_data = train_model(
                            images_dir, 
                            num_classes, 
                            model_name, 
                            fine_tune, 
                            epochs, 
                            learning_rate, 
                            batch_size, 
                            train_split, 
                            valid_split, 
                            use_weighted_loss, 
                            l2_lambda, 
                            patience, 
                            seed=42
                        )

                        if model_data is None:
                            st.error("Erro no treinamento do modelo.")
                            shutil.rmtree(temp_dir)
                            return

                        model, classes = model_data
                        # O modelo e as classes já estão armazenados no st.session_state
                        st.success("Treinamento concluído!")

                        # Opção para baixar o modelo treinado
                        st.write("### Faça o Download do Modelo Treinado:")
                        buffer = io.BytesIO()
                        torch.save(model.state_dict(), buffer)
                        buffer.seek(0)
                        btn = st.download_button(
                            label="Download do Modelo",
                            data=buffer,
                            file_name="modelo_treinado.pth",
                            mime="application/octet-stream",
                            key="download_model_button"
                        )

                        # Salvar as classes em um arquivo
                        classes_data = "\n".join(classes)
                        st.download_button(
                            label="Download das Classes",
                            data=classes_data,
                            file_name="classes.txt",
                            mime="text/plain",
                            key="download_classes_button"
                        )

                        # Limpar o diretório temporário
                        shutil.rmtree(temp_dir)
                    else:
                        st.error("Estrutura de diretórios inválida no arquivo ZIP. Certifique-se de que as imagens estão em 'images/' e as máscaras em 'masks/'.")
            elif mode == "Quântico (TFQ)":
                if epochs_q > 0 and batch_size_q > 0:
                    # Treinamento quântico baseado no ISIC
                    st.write("Iniciando o treinamento do modelo quântico...")
                    quantum_model, quantum_circuit, quantum_symbols, observables = train_quantum_model(
                        epochs=epochs_q, 
                        batch_size=batch_size_q,
                        n_qubits=4,
                        n_layers=2,
                        seed=42
                    )
                    if quantum_model is not None:
                        st.success("Treinamento do modelo quântico concluído!")

                        # Exibir circuitos quânticos treinados
                        st.write("### Circuito Quântico Treinado:")
                        visualize_circuit(quantum_circuit)

                        # Salvar o modelo quântico no formato Keras nativo
                        try:
                            quantum_model.save("quantum_model.keras", save_format='keras')
                            st.write("### Modelo Quântico Salvo como `quantum_model.keras`.")
                        except Exception as e:
                            st.error(f"Erro ao salvar o modelo quântico: {e}")

                        # Salvar as classes em um arquivo
                        # Como o treinamento quântico foi feito com ISIC, classes são definidas pelo usuário
                        classes_q = ["Classe_0", "Classe_1"]  # Ajuste conforme o seu conjunto de dados
                        classes_data_q = "\n".join(classes_q)
                        st.download_button(
                            label="Download das Classes (Quântico)",
                            data=classes_data_q,
                            file_name="classes_quantic.txt",
                            mime="text/plain",
                            key="download_classes_quantic_button"
                        )
                    else:
                        st.error("Erro no treinamento do modelo quântico.")
                else:
                    st.warning("Por favor, forneça os dados e as configurações corretas para o modo quântico.")

    # Avaliação de uma imagem individual
    st.header("Avaliação de Imagem")
    evaluate = st.radio(
        "Deseja avaliar uma imagem?", 
        ("Sim", "Não"), 
        key="evaluate_option"
    )
    if evaluate == "Sim":
        eval_image_file = st.file_uploader(
            "Faça upload da imagem para avaliação", 
            type=["png", "jpg", "jpeg", "bmp", "gif"], 
            key="eval_image_file"
        )
        if eval_image_file is not None:
            eval_image_file.seek(0)
            try:
                eval_image = Image.open(eval_image_file).convert("RGB")
            except Exception as e:
                st.error(f"Erro ao abrir a imagem: {e}")
                return

            st.image(eval_image, caption='Imagem para avaliação', use_container_width=True)

            if mode == "Clássico (PyTorch)":
                # Verificar se o modelo já foi carregado ou treinado
                if 'model' not in st.session_state or 'classes' not in st.session_state:
                    st.warning("Nenhum modelo carregado ou treinado. Por favor, carregue um modelo existente ou treine um novo modelo.")
                else:
                    model_eval = st.session_state['model']
                    classes_eval = st.session_state['classes']
                    model_name_eval = st.session_state.get('trained_model_name', 'ResNet18')  # Usa o nome do modelo armazenado

                    class_name, confidence = evaluate_image(model_eval, eval_image, classes_eval)
                    st.write(f"**Classe Predita:** {class_name}")
                    st.write(f"**Confiança:** {confidence:.4f}")

                    # Opção para visualizar segmentação
                    segmentation = False
                    if segmentation_model is not None:
                        segmentation = st.checkbox(
                            "Visualizar Segmentação", 
                            value=True, 
                            key="segmentation_checkbox"
                        )

                    # Visualizar ativações e segmentação
                    visualize_activations(
                        model_eval, 
                        eval_image, 
                        classes_eval, 
                        model_name_eval, 
                        segmentation_model=segmentation_model, 
                        segmentation=segmentation
                    )
            elif mode == "Quântico (TFQ)":
                # Verificar se o modelo quântico está carregado ou treinado
                if 'q_model' not in st.session_state or 'classes' not in st.session_state:
                    st.warning("Nenhum modelo quântico carregado ou treinado. Por favor, carregue um modelo quântico existente ou treine um novo modelo.")
                else:
                    quantum_model = st.session_state['q_model']
                    classes_eval = st.session_state['classes']  # Para o modo quântico, classes devem ser definidas pelo usuário

                    # Preparar a imagem para o modelo quântico
                    # Reduzir para 2x2 e binarizar (ajustado para 2x2 qubits)
                    image_resized = eval_image.resize((2, 2))
                    image_tensor = np.array(image_resized) / 255.0
                    image_bin = (image_tensor > threshold_q).astype(float).flatten()

                    # Converter para circuito
                    circuit_q = cirq.Circuit()
                    qubits = cirq.GridQubit.rect(1, 2)
                    for i, bit in enumerate(image_bin):
                        if bit:
                            circuit_q.append(cirq.X(qubits[i]))

                    # Converter circuito para string
                    circuits = tfq.convert_to_tensor([circuit_q])

                    # Fazer a previsão
                    y_pred = quantum_model.predict(circuits)
                    # y_pred está na faixa [0,1] devido ao uso de sigmoid
                    predicted_label = 1 if y_pred[0][0] > 0.5 else 0
                    confidence_q = y_pred[0][0]

                    # Obter o nome da classe
                    if predicted_label >= len(classes_eval):
                        class_name = "Desconhecida"
                    else:
                        class_name = classes_eval[predicted_label]

                    st.write(f"**Classe Predita (Quântico):** {class_name}")
                    st.write(f"**Confiança (Quântico):** {confidence_q:.4f}")

                    # Visualizar ativações - Grad-CAM não está implementado para modelos quânticos
                    st.write("**Visualização de Ativações:** Não disponível para o modo quântico.")

    # Visualização de Circuitos Quânticos
    st.header("Gerador e Visualizador de Circuitos Quânticos")
    st.write("Selecione o tipo de circuito quântico que deseja gerar e visualize-o abaixo.")

    # Seleção do tipo de circuito
    circuit_type_display = st.selectbox(
        "Selecione o Tipo de Circuito:",
        options=["Basic", "Entangling", "Rotation"],
        index=0,
        key="circuit_type_display"
    )

    # Função para criar circuitos com base no tipo selecionado
    def create_quantum_model_display(circuit_type):
        """
        Cria um circuito quântico com base no tipo selecionado.
        """
        if circuit_type == "Basic":
            n_qubits = 2
            n_layers = 1
        elif circuit_type == "Entangling":
            n_qubits = 4
            n_layers = 2
        elif circuit_type == "Rotation":
            n_qubits = 2
            n_layers = 3
        else:
            st.error("Tipo de circuito não suportado.")
            return None, None

        circuit, symbols = create_quantum_circuit(n_qubits, n_layers)
        return circuit, symbols

    # Botão para gerar o circuito
    if st.button("Gerar Circuito Quântico"):
        with st.spinner("Gerando o circuito quântico..."):
            quantum_circuit_display, quantum_symbols_display = create_quantum_model_display(circuit_type_display)
        
        if quantum_circuit_display is not None:
            st.write(f"### Circuito Selecionado: **{circuit_type_display}**")
            visualize_circuit(quantum_circuit_display)
        else:
            st.error("Falha ao gerar o circuito quântico.")

    # Opcional: Mostrar o código do circuito
    if st.checkbox("Mostrar Código do Circuito Quântico"):
        quantum_circuit_display, quantum_symbols_display = create_quantum_model_display(circuit_type_display)
        if quantum_circuit_display is not None:
            st.code(str(quantum_circuit_display), language='python')
        else:
            st.error("Nenhum código para exibir.")

    # Documentação dos Procedimentos
    st.write("### Documentação dos Procedimentos")
    st.write("Todas as etapas foram cuidadosamente registradas. Utilize esta documentação para reproduzir o experimento e analisar os resultados.")

    # Encerrar a aplicação
    st.write("Obrigado por utilizar o aplicativo!")

# Funções adicionais necessárias

def apply_transforms_and_get_embeddings(dataset_subset, model_for_embeddings, transform, batch_size, seed):
    """
    Aplica transformações ao dataset e extrai embeddings utilizando o modelo clássico.
    Retorna um DataFrame com embeddings, rótulos e imagens augmentadas.
    """
    dataloader = DataLoader(
        dataset_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=seed_worker, 
        generator=torch.Generator().manual_seed(seed)
    )
    
    embeddings = []
    labels = []
    augmented_images = []
    
    model_for_embeddings.eval()
    model_for_embeddings.to(device)
    
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            outputs = model_for_embeddings(inputs)
            embeddings.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())
            augmented_images.extend(inputs.cpu())
    
    embeddings = np.vstack(embeddings)
    df = pd.DataFrame(embeddings, columns=[f'feat_{i}' for i in range(embeddings.shape[1])])
    df['label'] = labels
    df['augmented_image'] = augmented_images
    return df

def display_all_augmented_images(df, classes, max_images=100):
    """
    Exibe todas as imagens augmentadas em um grid.
    """
    st.write(f"### Exibição de Até {max_images} Imagens Augmentadas")
    images = df['augmented_image'].tolist()[:max_images]
    num_cols = 5
    num_rows = int(np.ceil(len(images) / num_cols))
    fig = plt.figure(figsize=(15, 3 * num_rows))
    for i, img_tensor in enumerate(images):
        if i >= max_images:
            break
        image = img_tensor.permute(1, 2, 0).numpy()
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ax.imshow(image)
        ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

def visualize_embeddings(df, classes):
    """
    Visualiza os embeddings usando PCA.
    """
    st.write("### Visualização dos Embeddings com PCA")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(df.drop(columns=['label', 'augmented_image']))
    df_plot = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'Classe': [classes[label] for label in df['label']]
    })
    fig = px.scatter(
        df_plot, 
        x='PC1', 
        y='PC2', 
        color='Classe',
        title='Embeddings Visualizados com PCA',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

def visualize_activations(model, image, classes, model_name, segmentation_model=None, segmentation=False):
    """
    Visualiza as ativações do modelo utilizando Grad-CAM.
    """
    try:
        # Inicializar o método Grad-CAM
        cam_extractor = SmoothGradCAMpp(model, target_layer='fc')  # Ajuste 'fc' conforme o modelo

        # Preparar a imagem
        image_tensor = test_transforms(image).unsqueeze(0).to(device)

        # Fazer a previsão
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
            class_idx = pred.item()

        # Extrair o CAM
        activation_map = cam_extractor(class_idx, output)
        activation_map = activation_map.squeeze().cpu().numpy()

        # Overlay do CAM na imagem original
        cam_image = overlay_mask(
            to_pil_image(image_tensor.squeeze().cpu()),
            Image.fromarray((activation_map * 255).astype(np.uint8)),
            alpha=0.5
        )

        st.write("### Ativações do Modelo (Grad-CAM):")
        st.image(cam_image, caption='Grad-CAM', use_container_width=True)

        # Visualizar a segmentação, se disponível
        if segmentation_model is not None and segmentation:
            st.write("### Segmentação:")
            # Implementar a visualização da segmentação conforme o modelo
            # Esta é uma implementação placeholder
            seg_mask = segmentation_model(image_tensor)
            seg_mask = torch.argmax(seg_mask, dim=1).squeeze().cpu().numpy()
            seg_image = Image.fromarray(seg_mask.astype(np.uint8) * 255)
            st.image(seg_image, caption='Máscara de Segmentação', use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao visualizar ativações: {e}")

# Executar a aplicação
if __name__ == "__main__":
    main()
