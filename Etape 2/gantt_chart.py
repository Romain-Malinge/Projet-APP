import matplotlib.pyplot as plt
from collections import defaultdict


def plot_gantt(events, title="Gantt Chart", xlabel="Frame Number"):
    """
    Plot a Gantt chart from a list of events.

    Parameters
    ----------
    events : list of dict
        Each dict must contain:
        - 'category'
        - 'frame_start'
        - 'frame_end'
    """
    # Group events by category
    grouped = defaultdict(list)
    for event in events:
        grouped[event["category"]].append(
            (event["frame_start"], event["frame_end"])
        )

    categories = list(grouped.keys())
    y_positions = range(len(categories))

    fig, ax = plt.subplots(figsize=(12, 5))

    for y, category in zip(y_positions, categories):
        for start, end in grouped[category]:
            ax.barh(
                y,
                end - start,
                left=start,
                height=0.6
            )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def test_gantt():
    """Test function with sample data"""
    events = [
        {"category": "voiture", "frame_start": 80, "frame_end": 120},
        {"category": "voiture", "frame_start": 200, "frame_end": 260},
        {"category": "piéton", "frame_start": 90, "frame_end": 105},
        {"category": "piéton", "frame_start": 300, "frame_end": 320},
        {"category": "route", "frame_start": 250, "frame_end": 300},
        {"category": "panneau", "frame_start": 120, "frame_end": 180},
        {"category": "moto", "frame_start": 230, "frame_end": 240},
    ]

    plot_gantt(
        events,
        title="Example Gantt Chart from Frame Events",
        xlabel="Frame Index"
    )


def main():
    test_gantt()


if __name__ == "__main__":
    main()
