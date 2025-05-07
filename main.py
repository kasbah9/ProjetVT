# ECG Anomaly Detection - Google Colab Notebook (Amélioré avec les modules dédiés)

# --- Installation des bibliothèques nécessaires
!pip install wfdb
import wfdb
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import Counter

# --- Montage Google Drive (si besoin)
from google.colab import drive
drive.mount('/content/drive')

# --- Téléchargement des données si non présentes
import wfdb
wfdb.dl_database('mitdb', dl_dir='mitdb')

# --- Chargement d’un enregistrement et extraction des fenêtres
record_name = '100'
record = wfdb.rdrecord(os.path.join('mitdb', record_name))
annotation = wfdb.rdann(os.path.join('mitdb', record_name), 'atr')

window_size = 200
half_window = window_size // 2
signal = record.p_signal[:, 0]

X, y = [], []
for i, pos in enumerate(annotation.sample):
    if pos - half_window < 0 or pos + half_window > len(signal):
        continue
    beat = signal[pos - half_window : pos + half_window]
    label = 0 if annotation.symbol[i] == 'N' else 1
    X.append(beat)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(f"{len(X)} fenêtres extraites. Répartition : {Counter(y)}")

# --- Normalisation et split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# --- Modèle simple (MLP)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# --- Courbe d'apprentissage
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Courbe d'apprentissage")
plt.xlabel("Épochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# --- Évaluation finale
loss, acc = model.evaluate(X_test, y_test)
print(f"\nAccuracy sur test : {acc:.2f}")
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# --- Fonction de prédiction et visualisation (issue de analyse_record.py)
def predict_ecg(record_name, model, scaler, data_dir='mitdb', window_size=200):
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    annotation = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
    signal = record.p_signal[:, 0]
    half_window = window_size // 2

    X_pred, labels, positions = [], [], []
    for i, pos in enumerate(annotation.sample):
        if pos - half_window < 0 or pos + half_window > len(signal):
            continue
        beat = signal[pos - half_window : pos + half_window]
        label = 0 if annotation.symbol[i] == 'N' else 1
        X_pred.append(beat)
        labels.append(label)
        positions.append(pos)

    X_pred = np.array(X_pred)
    X_scaled = scaler.transform(X_pred)
    y_hat = (model.predict(X_scaled) > 0.5).astype(int).flatten()

    count = Counter(y_hat)
    print(f"\n📊 Résumé des prédictions pour {record_name} :")
    print(f"Normaux : {count[0]}, Anormaux : {count[1]}, Total : {len(y_hat)}")

    # Visualisation
    plt.figure(figsize=(15, 3))
    plt.plot(signal, alpha=0.6, label='Signal ECG')
    anomaly_positions = [positions[i] for i, pred in enumerate(y_hat) if pred == 1]
    plt.scatter(anomaly_positions, [signal[p] for p in anomaly_positions], color='red', s=10, label='Anomalies détectées')
    plt.title(f"Détection d'anomalies – Record {record_name}")
    plt.xlabel("Échantillons")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.grid()
    plt.show()

# --- Exécution sur un autre enregistrement (modifiable ici)
predict_ecg('111', model, scaler)
