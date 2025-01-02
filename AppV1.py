import streamlit as st
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import zipfile
import tempfile
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Configurações gerais
st.set_page_config(page_title="Quantum Image Classification", layout="wide")

# Funções auxiliares
def create_quantum_circuit(n_qubits, n_layers):
    """Cria um circuito quântico parametrizado."""
    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuit = cirq.Circuit()
    symbols = sympy.symbols(f'x0:{n_qubits * n_layers}')

    for layer in range(n_layers):
        for i in range(n_qubits):
            circuit.append(cirq.ry(symbols[layer * n_qubits + i])(qubits[i]))
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    return circuit, symbols

def build_quantum_model(n_qubits, n_layers):
    """Constrói o modelo quântico com TFQ."""
    circuit, symbols = create_quantum_circuit(n_qubits, n_layers)
    qubits = cirq.GridQubit.rect(1, n_qubits)
    observables = [cirq.Z(qubit) for qubit in qubits]

    inputs = tf.keras.Input(shape=(), dtype=tf.string)
    pqc = tfq.layers.PQC(circuit, observables)(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(pqc)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model, circuit, symbols

def prepare_quantum_data(features, n_qubits):
    """Prepara os dados para o modelo quântico aplicando PCA e escalonamento."""
    pca = PCA(n_components=n_qubits)
    features_pca = pca.fit_transform(features)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    features_scaled = scaler.fit_transform(features_pca)

    return features_scaled, pca, scaler

def convert_features_to_circuits(features, circuit, symbols):
    """Converte as features em tensores quânticos."""
    circuits = []
    for feature in features:
        resolved_circuit = cirq.resolve_parameters(circuit, dict(zip(symbols, feature)))
        circuits.append(resolved_circuit)

    return tfq.convert_to_tensor(circuits)

# Interface do Streamlit
st.title("Classificação Quântica de Imagens")
st.write("Treine um modelo quântico usando TensorFlow Quantum com dados de imagens.")

n_qubits = st.sidebar.slider("Número de Qubits", min_value=2, max_value=10, value=4)
n_layers = st.sidebar.slider("Número de Camadas", min_value=1, max_value=5, value=2)
epochs = st.sidebar.slider("Épocas", min_value=1, max_value=20, value=5)
batch_size = st.sidebar.selectbox("Tamanho do Lote", options=[4, 8, 16, 32], index=1)
learning_rate = st.sidebar.select_slider("Taxa de Aprendizagem", options=[0.01, 0.001, 0.0001], value=0.001)

uploaded_file = st.file_uploader("Envie um arquivo ZIP contendo as imagens", type="zip")

if uploaded_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        images_dir = os.path.join(temp_dir, 'images')
        if not os.path.exists(images_dir):
            st.error("O arquivo ZIP deve conter um diretório chamado 'images/'.")
        else:
            st.success("Imagens carregadas com sucesso!")

            # Simulação de extração de features
            num_samples = 100
            num_features = 20
            features = np.random.rand(num_samples, num_features)
            labels = np.random.randint(0, 2, size=num_samples)

            features_scaled, pca, scaler = prepare_quantum_data(features, n_qubits)
            quantum_model, circuit, symbols = build_quantum_model(n_qubits, n_layers)

            circuits_tensor = convert_features_to_circuits(features_scaled, circuit, symbols)

            quantum_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            st.write("### Treinando o modelo quântico...")
            history = quantum_model.fit(
                circuits_tensor,
                labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2
            )

            st.write("### Resultados do Treinamento")
            st.line_chart(history.history['accuracy'])
            st.line_chart(history.history['loss'])
