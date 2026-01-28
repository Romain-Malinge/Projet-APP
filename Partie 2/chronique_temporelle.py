import matplotlib.pyplot as plt
import numpy as np
from torch import threshold

class ChroniqueTemporelle:
    def __init__(self, labels):
        """
        Initialise la chronique avec des labels définis à l'avance.

        labels : liste de strings
        Exemple : ['voiture', 'piéton', 'route']
        """
        self.labels = list(labels)
        self.data = {label: [] for label in self.labels}
        self.ordered_data = []


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

    def extraire_stats(self, threshold):
        """
        Calcule les statistiques précises par classe.
        La durée d'un segment est définie par le nombre de frames réelles le composant.
        """
        if not hasattr(self, 'ordered_data') or not self.ordered_data:
            return {}

        # 1. Filtrage du bruit (flicker removal)
        data = self.ordered_data
        filtered = [data[i] for i in range(len(data)) if not (0 < i < len(data) - 1 
                    and data[i][0] != data[i+1][0] and data[i-1][0] == data[i+1][0])]
        
        total_frames_count = len(filtered)
        stats_per_label = {}

        for label in self.labels:
            # On récupère uniquement les numéros de frames pour ce label
            label_frames = [f for lbl, f in filtered if lbl == label]
            
            if not label_frames:
                stats_per_label[label] = {
                    'total_frames': 0, 'pourcentage': 0, 
                    'nb_segments': 0, 'moyenne_segment': 0
                }
                continue

            # 2. Identification des segments et comptage des frames par bloc
            segments_durations = []
            if label_frames:
                current_segment_size = 1
                
                for i in range(1, len(label_frames)):
                    # Si l'écart entre deux frames du même label est sous le seuil
                    if label_frames[i] - label_frames[i-1] <= threshold:
                        current_segment_size += 1
                    else:
                        # On enregistre le nombre de frames du segment terminé
                        segments_durations.append(current_segment_size)
                        current_segment_size = 1
                
                # Ne pas oublier le dernier segment
                segments_durations.append(current_segment_size)

            # 3. Calcul des metrics basées sur le comptage réel
            stats_per_label[label] = {
                'total_frames': len(label_frames),
                'pourcentage': (len(label_frames) / total_frames_count) * 100,
                'nb_segments': len(segments_durations),
                'moyenne_segment': np.mean(segments_durations),
                'max_segment': np.max(segments_durations)
            }

        # Affichage des résultats
        print(f"{'Label':<15} | {'Frames':<8} | {'%':<6} | {'Segments':<8} | {'Avg Frames':<10}")
        print("-" * 60)
        for label, s in stats_per_label.items():
            print(f"{label:<15} | {s['total_frames']:<8} | {s['pourcentage']:>5.1f}% | {s['nb_segments']:<8} | {s['moyenne_segment']:>9.1f}")
        
        return stats_per_label

    def afficher_more(self, titre="Analyse oculométrique sur frames labellisées", largeur_barre=0.6, threshold=10):
        """
        Affiche un graphique de barres horizontales (Gantt-style).
        Fusionne les frames consécutives du même label sous un seuil (threshold).
        """
        if not hasattr(self, 'ordered_data') or not self.ordered_data:
            print("Aucune donnée ordonnée à afficher.")
            return
        # 1. Groupement des données en segments (start, duration)
        # segments = { 'LabelA': [(start, width), (start, width)], ... }
        segments = {label: [] for label in self.labels}

        filtered_data = []
        if len(self.ordered_data) > 3:
            for i, data in enumerate(self.ordered_data):
                if i == 0 or i == len(self.ordered_data) - 1:
                    filtered_data.append(data)
                elif not (data[0] != self.ordered_data[i+1][0] and self.ordered_data[i-1][0] == self.ordered_data[i+1][0]):
                #elif not (data[0] != self.ordered_data[i-1][0] and data[0] != self.ordered_data[i+1][0]):
                    filtered_data.append(data)
        else:
            filtered_data = self.ordered_data.copy()
        
        data = filtered_data
        if data:
            current_label, current_start = data[0]
            prev_frame = current_start

            for label, frame in data[1:]:
                # Si même label et écart inférieur au seuil -> on continue le segment
                if label == current_label and (frame - prev_frame) <= threshold:
                    prev_frame = frame
                else:
                    # Sinon, on enregistre le segment précédent et on recommence
                    duration = prev_frame - current_start + 1
                    segments[current_label].append((current_start, duration))
                    
                    current_label = label
                    current_start = frame
                    prev_frame = frame
            
            # Ajout du dernier segment
            duration = prev_frame - current_start + 1
            segments[current_label].append((current_start, duration))

        # 2. Configuration du graphique
        fig, ax = plt.subplots(figsize=(12, max(6, len(self.labels) * 0.6)))
        couleurs = plt.cm.tab10(np.linspace(0, 1, len(self.labels)))
        label_to_color = {label: couleurs[i] for i, label in enumerate(self.labels)}

        all_frames = [f for _, f in self.ordered_data]
        
        # 3. Tracé des segments
        for i, label in enumerate(self.labels):
            label_segments = segments[label]
            if label_segments:
                # broken_barh est idéal pour des segments discontinus sur une même ligne
                ax.broken_barh(
                    label_segments, 
                    (i - largeur_barre/2, largeur_barre), 
                    facecolors=label_to_color[label],
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )

        # Mise en forme des axes
        ax.set_yticks(range(len(self.labels)))
        ax.set_yticklabels(self.labels)
        ax.set_xlabel('Numéro de Frame')
        ax.set_title(titre)

        if all_frames:
            min_f, max_f = min(all_frames), max(all_frames)
            marge = (max_f - min_f) * 0.05
            ax.set_xlim(min_f - marge, max_f + marge)

        ax.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig("chronique_temporelle_analyse.png")
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
                    self.ordered_data.extend((label, frame) for frame in frames)
                    self.data[label] = frames
                else:
                    print(f"Label inconnu dans le fichier : {label}")

            self.ordered_data.sort(key=lambda x: x[1] if x[1] else float('inf'))

if __name__ == "__main__":
    # Exemple d'utilisation (données approximatives basées sur l'image) [code:1]
    chrono = ChroniqueTemporelle(['ciel', "vegetation", 'pieton', 'route', 'panneau', 'batiment', "4 roues", "2 roues"])
    #chrono.ajouter_frames('voiture', [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 180, 240, 320, 380, 440, 500])
    #chrono.ajouter_frames('piéton', [110, 160, 220, 290, 360])
    #chrono.ajouter_frames('route', [130, 200, 270, 340])
    #chrono.ajouter_frames('panneau', [150, 210])
    #chrono.ajouter_frames('bâtiment', [170])
    chrono.load("./Partie 2/chronique_sujet1_start5_pas0.25.txt")
    chrono.extraire_stats(threshold=100)
    chrono.afficher_more(threshold=100)