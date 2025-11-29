import plotly.graph_objects as go
import numpy as np
from PIL import Image



def heat_map_density(x, y, W, H, distance=200):

    # Initialiser la matrice de densité
    z = np.zeros((H, W))

    # initialiser le masque de distance
    mask = np.zeros((distance, distance))

    # Remplir le masque avec des valeurs décroissantes en fonction de la distance au centre
    center = distance // 2
    for i in range(distance):
        for j in range(distance):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if dist < center:
                mask[i, j] = (center - dist) / center

    # Ajouter la contribution de chaque point de regard à la matrice de densité
    for i in range(len(x)):
        x_pos = int(x[i])
        y_pos = int(y[i])

        # Définir les limites pour appliquer le masque
        x_start = max(0, x_pos - distance // 2)
        x_end = min(W, x_pos + distance // 2)
        y_start = max(0, y_pos - distance // 2)
        y_end = min(H, y_pos + distance // 2)

        # Définir les limites du masque à appliquer
        mask_x_start = max(0, distance // 2 - x_pos)
        mask_x_end = mask_x_start + (x_end - x_start)
        mask_y_start = max(0, distance // 2 - y_pos)
        mask_y_end = mask_y_start + (y_end - y_start)

        # Ajouter le masque à la matrice de densité
        z[y_start:y_end, x_start:x_end] += mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

    return z



def show_points_on_poster(affiche_path, x, y, plot_W=400):

    # Changer le chemin de l'image selon le nom de l'affiche
    poster = Image.open(affiche_path)
    W, H = poster.size  # dimensions réelles de l'image
    n_points = len(x)

    # Création du graphe
    fig = go.Figure()

    # Ajouter l'image en dessous
    fig.add_layout_image(
        dict(
            source=poster,
            x=0,
            y=H,
            sizex=W,
            sizey=H,
            xref="x",
            yref="y",
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Créer des frames pour l'animation
    frames = []
    for i in range(1, n_points + 1):
        frame = go.Frame(
            data=[
                go.Scatter(x=x[:i], y=y[:i], mode='markers', marker=dict(color='red', size=6)),
                go.Scatter(x=x[:i], y=y[:i], mode='lines', line=dict(color='red', width=2), hoverinfo='skip')
            ],
            name=str(i)
        )
        frames.append(frame)

    # Ajouter le premier point/ligne
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='markers', marker=dict(color='red', size=6)))
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode='lines', line=dict(color='red', width=2), hoverinfo='skip'))

    # Ajouter les frames à la figure
    fig.frames = frames

    # Boutons d'animation en haut à gauche
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1,
            x=0,
            xanchor="left",
            yanchor="top",
            pad=dict(t=0, l=10),
            buttons=[dict(label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=5, redraw=True), fromcurrent=True, mode='immediate')])]
        )]
    )

    # Ajuster les axes pour correspondre à l'image
    fig.update_xaxes(range=[0, W], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[0, H], showgrid=False, zeroline=False, visible=False, scaleanchor="x")

    # Rendu propre : enlever marges
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        width=plot_W,
        height=int(plot_W * (H / W)),
        showlegend=False)
    
    fig.show()



def show_heat_map_on_poster(affiche_path, z, plot_W=400):

    # Changer le chemin de l'image selon le nom de l'affiche
    poster = Image.open(affiche_path)
    W, H = poster.size  # dimensions réelles de l'image

    # Création du graphe
    fig = go.Figure()

    # Ajouter l'image en dessous
    fig.add_layout_image(
        dict(
            source=poster,
            x=0,
            y=H,
            sizex=W,
            sizey=H,
            xref="x",
            yref="y",
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    fig.add_trace(go.Heatmap(
        z=z,
        opacity=1,
        showlegend=False,
        hoverinfo='skip',
        showscale=False,
        colorscale=[
            [0.0, "rgba(255,255,255,0.0)"],
            [0.2, "rgba(0,0,255,0.4)"],
            [0.4, "rgba(0,255,0,0.9)"],
            [0.6, "rgba(255,255,0,0.9)"],
            [0.8, "rgba(255,165,0,0.9)"],
            [1.0, "rgba(255,0,0,0.9)"]
        ]
    ))

    # Ajuster les axes pour correspondre à l'image
    fig.update_xaxes(range=[0, W], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[0, H], showgrid=False, zeroline=False, visible=False, scaleanchor="x")

    # Rendu propre : enlever marges
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        width=plot_W,
        height=int(plot_W * (H / W)),
        showlegend=False)

    fig.show()