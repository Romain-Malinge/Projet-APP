import matplotlib.pyplot as plt
import numpy as np

class ChroniqueTemporelle:
    def __init__(self):
        """
        Initialise la chronique avec une liste de labels et un dictionnaire de données.
        """
        self.labels = []
        self.data = {}  # label -> liste de frames (int)

    def ajouter_label(self, label, frames):
        """
        Ajoute un label et ses positions de frames (liste d'entiers).
        Exemple: ajouter_label('voiture', [100, 150, 200, 250, 300, 350, 400, 450, 500])
        """
        self.labels.append(label)
        self.data[label] = sorted(frames)

    def afficher(self, titre="Analyse oculométrique sur frames labellisées", largeur=0.8):
        """
        Affiche le graphique horizontal similaire à l'image.
        - Barre par label, positionnée aux frames indiquées.
        - Couleurs automatiques vertes/bleues par défaut.
        """
        if not self.data:
            print("Aucune donnée à afficher.")
            return

        y_pos = np.arange(len(self.labels))
        fig, ax = plt.subplots(figsize=(12, max(6, len(self.labels)*0.5)))

        couleurs = plt.cm.Set3(np.linspace(0, 1, len(self.labels)))

        for i, label in enumerate(self.labels):
            frames = self.data[label]
            if frames:
                x_centers = np.array(frames)
                ax.barh(i, largeur, left=x_centers - largeur/2, height=0.6,
                        label=label, color=couleurs[i], alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.labels)
        ax.set_xlabel('Numéro de Frame')
        ax.set_title(titre)
        ax.set_xlim(0, max(max(d) for d in self.data.values()) * 1.1 if self.data else 500)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Exemple d'utilisation (données approximatives basées sur l'image) [code:1]
    chrono = ChroniqueTemporelle()
    chrono.ajouter_label('voiture', [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 180, 240, 320, 380, 440, 500])
    chrono.ajouter_label('piéton', [110, 160, 220, 290, 360])
    chrono.ajouter_label('route', [130, 200, 270, 340])
    chrono.ajouter_label('panneau', [150, 210])
    chrono.ajouter_label('bâtiment', [170])
    chrono.afficher()