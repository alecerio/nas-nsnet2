import numpy as np

# Carica il file .npy
file_path = "/home/alessandro/Desktop/nas-nsnet2/nsnet2/pytorch/numpy_weights/fc1_bias.npy"  # Sostituisci con il percorso corretto del tuo file .npy
data = np.load(file_path)

# Stampa le informazioni principali
print(f"Tipo di dato: {data.dtype}")
print(f"Shape: {data.shape}")
print("Dati:")

# Stampa i dati del tensor
print(data)
