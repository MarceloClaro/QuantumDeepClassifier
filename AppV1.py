

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import io
import zipfile
import tempfile
import shutil
import os
import random
import logging
from datetime import datetime
import gc
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# Importações para TensorFlow Quantum
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy

# Configurar o dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurar o log
logging.basicConfig(level=logging.INFO)

# Funções Auxiliares

def set_seed(seed):
    """
    Define a seed para reprodução.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def seed_worker(worker_id):
    """
    Função para inicializar seeds nos workers do DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_model(model_name, num_classes, dropout_p=0.5, fine_tune=False, seed=42):
    """
    Carrega um modelo pré-treinado e ajusta a última camada para o número de classes.
    """
    set_seed(seed)
    try:
        if model_name == 'ResNet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'ResNet50':
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'DenseNet121':
            model = models.densenet121(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        else:
            st.error("Modelo pré-treinado não suportado.")
            return None

        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False

        model = model.to(device)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

def create_quantum_circuit(n_qubits, n_layers):
    """
    Cria um circuito quântico parametrizado.
    """
    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuit = cirq.Circuit()
    symbols = sympy.symbols(f'x0:{n_qubits * n_layers}')
    
    for layer in range(n_layers):
        for i in range(n_qubits):
            circuit.append(cirq.ry(symbols[layer * n_qubits + i])(qubits[i]))
        # Entangling layer
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    return circuit, symbols

def convert_features_to_circuits(features, circuit, symbols):
    """
    Converte features numéricas em circuitos quânticos aplicando os parâmetros.
    """
    circuits = []
    for feature in features:
        parameter_values = feature.tolist()
        parametrized_circuit = cirq.resolve_parameters(circuit, dict(zip(symbols, parameter_values)))
        circuits.append(parametrized_circuit)
    return tfq.convert_to_tensor(circuits)

def build_quantum_model(n_qubits, n_layers, symbols, observables):
    """
    Constrói o modelo quântico utilizando TensorFlow Quantum.
    """
    readout = tf.keras.layers.Dense(1, activation='sigmoid')
    inputs = tf.keras.Input(shape=(), dtype=tf.string)
    x = tfq.layers.PQC(tfq.convert_to_tensor([cirq.Circuit() for _ in range(len(symbols))]), observables)(inputs)
    outputs = readout(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def prepare_quantum_data(train_df, valid_df, test_df, n_qubits):
    """
    Aplica PCA nos embeddings e prepara os dados para o modelo quântico.
    
    Retorna os dados transformados e os objetos PCA e scaler.
    """
    # Combinar os dados de treino e validação para ajustar o PCA
    combined_features = np.vstack([train_df.drop(columns=['label', 'augmented_image']),
                                   valid_df.drop(columns=['label', 'augmented_image'])])
    
    # Aplicar PCA para reduzir a dimensionalidade para o número de qubits
    pca = PCA(n_components=n_qubits)
    pca.fit(combined_features)
    
    # Transformar os conjuntos de dados
    train_features_pca = pca.transform(train_df.drop(columns=['label', 'augmented_image']))
    valid_features_pca = pca.transform(valid_df.drop(columns=['label', 'augmented_image']))
    test_features_pca = pca.transform(test_df.drop(columns=['label', 'augmented_image']))
    
    # Escalar os dados para [0, π]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(train_features_pca)
    
    train_features_scaled = scaler.transform(train_features_pca)
    valid_features_scaled = scaler.transform(valid_features_pca)
    test_features_scaled = scaler.transform(test_features_pca)
    
    return train_features_scaled, valid_features_scaled, test_features_scaled, pca, scaler

def build_quantum_model(n_qubits, n_layers, symbols, observables):
    """
    Constrói o modelo quântico utilizando TensorFlow Quantum.
    """
    readout = tf.keras.layers.Dense(1, activation='sigmoid')
    inputs = tf.keras.Input(shape=(), dtype=tf.string)
    x = tfq.layers.PQC(tfq.convert_to_tensor([cirq.Circuit() for _ in range(len(symbols))]), observables)(inputs)
    outputs = readout(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_quantum_model(
    train_features_scaled, 
    train_labels, 
    valid_features_scaled, 
    valid_labels, 
    n_qubits, 
    n_layers, 
    epochs=10, 
    batch_size=32, 
    learning_rate=0.01, 
    optimizer_type='Adam',
    seed=42
):
    """
    Treina um modelo quântico utilizando TFQ.
    
    Retorna o modelo quântico treinado e o histórico de treinamento.
    """
    set_seed(seed)
    
    # Definir qubits e circuitos
    circuit, symbols = create_quantum_circuit(n_qubits, n_layers)
    observables = [cirq.Z(qubit) for qubit in cirq.GridQubit.rect(1, n_qubits)]
    
    # Construir modelo quântico
    quantum_model = build_quantum_model(n_qubits, n_layers, symbols, observables)
    
    # Selecionar o otimizador
    if optimizer_type == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_type == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        st.error("Tipo de otimizador não suportado.")
        return None, None
    
    # Compilar o modelo
    quantum_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Converter features para circuitos
    train_circuits = convert_features_to_circuits(train_features_scaled, circuit, symbols)
    valid_circuits = convert_features_to_circuits(valid_features_scaled, circuit, symbols)
    
    # Treinar o modelo
    history = quantum_model.fit(
        train_circuits,
        train_labels,
        validation_data=(valid_circuits, valid_labels),
        epochs=epochs,
        batch_size=batch_size
    )
    
    return quantum_model, history

def evaluate_image(model, image, classes, pca=None, scaler=None, quantum=False):
    """
    Avalia uma única imagem usando o modelo treinado.
    
    Se quantum=True, utiliza o modelo quântico com PCA e scaler.
    """
    if not quantum:
        # Modelo clássico
        model.eval()
        with torch.no_grad():
            image_tensor = test_transforms(image).unsqueeze(0).to(device)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            predicted_class = classes[preds.item()]
            confidence = confidence.item()
        return predicted_class, confidence
    else:
        # Modelo quântico
        if 'quantum_model' not in st.session_state or 'quantum_circuit' not in st.session_state or 'quantum_symbols' not in st.session_state:
            st.error("Modelo quântico não carregado corretamente.")
            return None, None
        quantum_model = st.session_state['quantum_model']
        quantum_circuit = st.session_state['quantum_circuit']
        quantum_symbols = st.session_state['quantum_symbols']
        pca = st.session_state['pca']
        scaler = st.session_state['scaler']
        
        # Extrair features com o modelo clássico
        with torch.no_grad():
            model_for_embeddings = st.session_state['model']
            model_for_embeddings.eval()
            image_tensor = test_transforms(image).unsqueeze(0).to(device)
            features = model_for_embeddings(image_tensor).cpu().numpy()
        
        # Aplicar PCA e Escalonamento
        features_pca = pca.transform(features)
        features_scaled = scaler.transform(features_pca)
        
        # Converter features para circuitos quânticos
        circuits = convert_features_to_circuits(features_scaled, quantum_circuit, quantum_symbols)
        
        # Fazer a previsão
        y_pred = quantum_model.predict(circuits)
        
        # Interpretar a saída
        predicted_label = int(y_pred[0][0] > 0.5)
        confidence = y_pred[0][0]
        
        # Obter o nome da classe
        if predicted_label >= len(classes):
            class_name = "Desconhecida"
        else:
            class_name = classes[predicted_label]
        
        return class_name, confidence

def visualize_data(dataset, classes):
    """
    Visualiza algumas imagens de cada classe.
    """
    st.write("### Visualização de Imagens das Classes")
    num_classes = len(classes)
    num_images = min(5, len(dataset.classes))
    fig, axes = plt.subplots(nrows=num_classes, ncols=num_images, figsize=(15, 3 * num_classes))
    for cls in range(num_classes):
        cls_idx = [i for i, label in enumerate(dataset.targets) if label == cls]
        sampled_idx = random.sample(cls_idx, min(num_images, len(cls_idx)))
        for i, idx in enumerate(sampled_idx):
            img, label = dataset[idx]
            img = img.permute(1, 2, 0).numpy()
            axes[cls, i].imshow(img)
            axes[cls, i].axis('off')
            if i == 0:
                axes[cls, i].set_ylabel(classes[cls], fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_class_distribution(dataset, classes):
    """
    Plota a distribuição de classes.
    """
    st.write("### Distribuição das Classes")
    class_counts = pd.Series(dataset.targets).value_counts().sort_index()
    df = pd.DataFrame({'Classe': classes, 'Contagem': class_counts.values})
    fig = px.bar(df, x='Classe', y='Contagem', title='Distribuição das Classes', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

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
        plt.close(fig)
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

def visualize_activations(model, image, classes, model_name, segmentation_model=None, segmentation=False):
    """
    Visualiza as ativações do modelo utilizando Grad-CAM.
    """
    try:
        # Inicializar o método Grad-CAM
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

        target_layer = 'layer4' if model_name.startswith('ResNet') else 'features'
        cam = GradCAM(model=model, target_layers=[getattr(model, target_layer)], use_cuda=torch.cuda.is_available())

        # Preparar a imagem
        image_tensor = test_transforms(image).unsqueeze(0).to(device)

        # Fazer a previsão
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            _, pred = torch.max(outputs, 1)
            target_category = pred.item()

        # Extrair o CAM
        grayscale_cam = cam(input_tensor=image_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(np.array(image.convert('RGB'))/255.0, grayscale_cam, use_rgb=True)

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

def train_segmentation_model(images_dir, masks_dir, num_classes):
    """
    Função placeholder para treinar um modelo de segmentação.
    Implementação completa depende dos requisitos específicos e dos dados.
    """
    st.write("**Treinamento do modelo de segmentação não implementado.**")
    # Aqui você deve implementar o treinamento do modelo de segmentação conforme necessário
    return None

# Função Principal

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
        segmentation_model = train_segmentation_model(None, None, num_classes_segmentation)
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
        n_qubits = st.sidebar.number_input(
            "Número de Qubits:", 
            min_value=2, 
            max_value=10, 
            value=4, 
            step=1, 
            key="n_qubits"
        )
        n_layers = st.sidebar.number_input(
            "Número de Camadas no Circuito Quântico:", 
            min_value=1, 
            max_value=10, 
            value=2, 
            step=1, 
            key="n_layers"
        )
        learning_rate_q = st.sidebar.select_slider(
            "Taxa de Aprendizagem (Quântico):", 
            options=[0.1, 0.01, 0.001, 0.0001], 
            value=0.01, 
            key="learning_rate_q"
        )
        optimizer_type_q = st.sidebar.selectbox(
            "Tipo de Otimizador (Quântico):", 
            options=['Adam', 'SGD', 'RMSprop'], 
            key="optimizer_type_q"
        )
        threshold_q = st.sidebar.slider(
            "Threshold para Binarização [0,1] (Quântico):", 
            0.0, 
            1.0, 
            0.5, 
            step=0.05, 
            key="threshold_q"
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
                    quantum_model = tf.keras.models.load_model(q_model_file, compile=False, custom_objects={'PQC': tfq.layers.PQC})
                    st.session_state['quantum_model'] = quantum_model
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
                            data_dir=images_dir, 
                            num_classes=num_classes, 
                            model_name=model_name, 
                            fine_tune=fine_tune, 
                            epochs=epochs, 
                            learning_rate=learning_rate, 
                            batch_size=batch_size, 
                            train_split=train_split, 
                            valid_split=valid_split, 
                            use_weighted_loss=use_weighted_loss, 
                            l2_lambda=l2_lambda, 
                            patience=patience, 
                            seed=42,
                            mode=mode
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
                if n_qubits >= 2 and n_qubits <= 10:
                    # Aqui você pode implementar o treinamento quântico com os dados fornecidos
                    st.write("**Treinamento do modelo quântico ainda não implementado para upload de dados.**")
                    # Placeholder para a funcionalidade de upload de dados quânticos
                else:
                    st.error("O número de qubits deve estar entre 2 e 10 para o treinamento quântico.")
    
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
                if 'quantum_model' not in st.session_state or 'classes' not in st.session_state or 'pca' not in st.session_state or 'scaler' not in st.session_state:
                    st.warning("Nenhum modelo quântico carregado ou treinado. Por favor, carregue um modelo quântico existente ou treine um novo modelo.")
                else:
                    quantum_model = st.session_state['quantum_model']
                    quantum_circuit = st.session_state['quantum_circuit']
                    quantum_symbols = st.session_state['quantum_symbols']
                    pca = st.session_state['pca']
                    scaler = st.session_state['scaler']
                    classes_eval = st.session_state['classes']  # Para o modo quântico, classes devem ser definidas pelo usuário

                    class_name, confidence_q = evaluate_image(model=None, image=eval_image, classes=classes_eval, pca=pca, scaler=scaler, quantum=True)
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
            st.write(quantum_circuit_display)
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

# Executar a aplicação
if __name__ == "__main__":
    main()
