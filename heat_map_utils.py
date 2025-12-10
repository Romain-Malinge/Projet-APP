import numpy as np

def get_test_set(W, H):
    
    # Simuler des données de regard (coordonnées x, y)
    n_points = 500
    x = []
    y = []
    x_cur = W // 2
    y_cur = H // 2

    for _ in range(n_points):
        dir = np.random.choice(['left', 'right', 'top', 'bottom'], p=[0.25, 0.25, 0.25, 0.25])
        pas = np.random.randint(50, 200)
        if dir == 'left':
            x_cur -= pas
        elif dir == 'right':
            x_cur += pas
        elif dir == 'top':
            y_cur -= pas  # vers le haut de l'image
        else:  # bottom
            y_cur += pas  # vers le bas de l'image
        
        x.append(x_cur)
        y.append(y_cur)

    x = np.array(x)
    y = np.array(y)

    # Inverser y pour Plotly (origine en bas à gauche)
    y = H - y

    return x, y



def traitement_points(x, y, W, H):
    # S'assurer que les points sont dans les limites de l'image
    # et ajouter un NaN pour les points hors limites
    x_valid = []
    y_valid = []
    for i in range(len(x)):
        if 0 <= x[i] < W and 0 <= y[i] < H:
            x_valid.append(x[i])
            y_valid.append(y[i])
        else:
            if len(x_valid) >= 1 and x_valid[-1] != np.nan:
                x_valid.append(np.nan)
                y_valid.append(np.nan)
    
    return x, y