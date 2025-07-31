import matplotlib.pyplot as plt

def plot_simulation(steps, temps, currents, actions, cache_hits):
    action_colors = {
        "cut_power": "red",
        "cool_down": "orange",
        "limit_current": "blue",
        "increase_current": "purple",
        "normal": "green"
    }

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Sıcaklık çizimi
    ax1.plot(steps, temps, color="black", label="Sıcaklık (°C)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Sıcaklık (°C)", color="black")

    if len(steps) < 1000:
        # Cache hit gösterimi
        for i, hit in enumerate(cache_hits):
            if hit:
                ax1.scatter(steps[i], temps[i], color="black", s=100, edgecolors='k', label="Cache Hit" if i == 0 else "")

        # Aksiyon renkleri
        for i, action in enumerate(actions):
            ax1.scatter(steps[i], temps[i], color=action_colors.get(action, "gray"), s=40, alpha=0.7, label=action if i == 0 else "")

    # Her aksiyon için legend örneği
    for action, color in action_colors.items():
        if action == "normal":
            continue;
        ax1.scatter([], [], color=color, s=40, label=action)


    # Akım için ikinci y ekseni
    ax2 = ax1.twinx()
    ax2.plot(steps, currents, color="blue", linestyle="--", label="Akım (A)")
    ax2.set_ylabel("Akım (A)", color="blue")

    # Akım ve sıcaklık için referans aralıklarını gösteren çizgiler
    ax1.axhline(y=25, color="red", linestyle=":", linewidth=2 )
    ax1.axhline(y=40, color="red", linestyle=":", linewidth=2)
    ax2.axhline(y=100, color="purple", linestyle=":", linewidth=2)
    ax2.axhline(y=250, color="purple", linestyle=":", linewidth=2)

    # Legend ayarı
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.title("EV Battery Reflex Simulation")
    plt.grid(True)
    plt.show()