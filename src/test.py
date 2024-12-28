import numpy as np
import rerun as rr

# Inizializza Rerun
rr.init("Landmark Plot - Example Data", spawn=True)

# Funzione per leggere il file world.dat
def read_world_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ignora le righe vuote
                parts = line.split()
                landmark_id = int(parts[0])  # ID del landmark
                position = list(map(float, parts[1:4]))  # Posizione (x, y, z)
                appearance = list(map(float, parts[4:]))  # Numeri associati all'aspetto
                data.append((landmark_id, position, appearance))
    return data

# Funzione per controllare la validità di una posizione
def is_valid_position(position):
    return all(np.isfinite(coord) for coord in position)

# Percorso al file world.dat
file_path = "../data/world.dat"

# Leggi i dati
landmarks = read_world_file(file_path)

# Filtra i punti con posizioni valide
valid_landmarks = [lm for lm in landmarks if is_valid_position(lm[1])]

# Estrarre i dati per il logging
positions = np.array([lm[1] for lm in valid_landmarks])  # Posizioni valide (x, y, z)


rr.log(
    "landmarks",  # Path dell'entità
    rr.Points3D(positions, colors=(255,0,0), radii=0.02)
)
