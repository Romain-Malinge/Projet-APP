import matplotlib.pyplot as plt
import numpy as np

class ChroniqueTemporelle:
    def __init__(self, labels):
        """
        Initialise la chronique avec des labels définis à l'avance.

        labels : liste de strings
        Exemple : ['voiture', 'piéton', 'route']
        """
        self.labels = list(labels)
        self.data = {label: [] for label in self.labels}

    def ajouter_frame(self, label, frame):
        """
        Ajoute une frame (int) pour un label donné.
        """
        if label not in self.data:
            raise ValueError(f"Label inconnu : {label}")
        self.data[label].append(frame)

    def ajouter_frames(self, label, frames):
        """
        Ajoute plusieurs frames pour un label donné.
        """
        if label not in self.data:
            raise ValueError(f"Label inconnu : {label}")
        self.data[label].extend(frames)

    def afficher(self, titre="Analyse oculométrique sur frames labellisées", largeur=0.8):
        """
        Affiche le graphique horizontal similaire à l'image.
        - Barre par label, positionnée aux frames indiquées.
        - Axe x démarrant à la première frame annotée.
        """
        if not self.data:
            print("Aucune donnée à afficher.")
            return

        y_pos = np.arange(len(self.labels))
        fig, ax = plt.subplots(figsize=(12, max(6, len(self.labels)*0.5)))

        couleurs = plt.cm.tab10(np.linspace(0, 1, len(self.labels)))

        all_frames = []

        for i, label in enumerate(self.labels):
            frames = self.data[label]
            if frames:
                frames = np.array(frames)
                all_frames.extend(frames)
                ax.barh(
                    i,
                    largeur,
                    left=frames - largeur / 2,
                    height=0.6,
                    color=couleurs[i],
                    alpha=0.9
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.labels)
        ax.set_xlabel('Numéro de Frame')
        ax.set_title(titre)

        if all_frames:
            min_frame = min(all_frames)
            max_frame = max(all_frames)
            marge = (max_frame - min_frame) * 0.05  # 5% de marge visuelle
            ax.set_xlim(min_frame - marge, max_frame + marge)

        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save(self, filepath):
        """
        Sauvegarde les données de la chronique dans un fichier texte.
        filepath : chemin du fichier de sauvegarde
        """
        with open(filepath, 'w') as f:
            for label, frames in self.data.items():
                f.write(f"{label}: {frames}\n")

    def load(self, filepath):
        """
        Charge les données de la chronique depuis un fichier texte.
        filepath : chemin du fichier de chargement
        """
        with open(filepath, 'r') as f:
            for line in f:
                label, frames_str = line.strip().split(': ')
                frames = eval(frames_str)
                if label in self.data:
                    self.data[label] = frames
                else:
                    print(f"Label inconnu dans le fichier : {label}")

if __name__ == "__main__":
    # Exemple d'utilisation (données approximatives basées sur l'image) [code:1]
    chrono = ChroniqueTemporelle(['voiture', 'piéton', 'route', 'panneau', 'bâtiment'])
    chrono.ajouter_frames('voiture', [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 180, 240, 320, 380, 440, 500])
    chrono.ajouter_frames('piéton', [110, 160, 220, 290, 360])
    chrono.ajouter_frames('route', [130, 200, 270, 340])
    chrono.ajouter_frames('panneau', [150, 210])
    chrono.ajouter_frames('bâtiment', [170])
    # chrono.save("chronique_exemple.txt")
    chrono.afficher()