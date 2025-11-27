import plotly.graph_objects as go
from PIL import Image


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
            opacity=0.8,
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
        showlegend=False
    )

    fig.show()



def show_it_map_on_poster(affiche_path, z, plot_W=500):
    
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
            opacity=0.8,
            layer="below"
        )
    )

    # Ajouter la heatmap
    fig.add_trace(go.Heatmap(z=z))

    # Ajuster les axes pour correspondre à l'image
    fig.update_xaxes(range=[0, W], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[0, H], showgrid=False, zeroline=False, visible=False, scaleanchor="x")

    # Rendu propre : enlever marges
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        width=plot_W,
        height=int(plot_W * (H / W)),
        showlegend=False
    )

    fig.show()