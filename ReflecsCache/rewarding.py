

def evaluate_reward(state, action):
    temp = state["cell_temp"]
    current = state["current_draw"]
    voltage = state["voltage"]
    ambient = state["ambient_temp"]
    charging = state["charging"]

    danger = False
    mild = False

    if temp > 100 or (charging and temp > 90):
        danger = True
    if voltage > 390 or current > 180 or temp > 80:
        mild = True

    # Reward logic
    if danger and action == "cut_power":
        return +1
    elif danger and action != "cut_power":
        return -1
    elif mild and action in ["cool_down", "limit_current"]:
        return +1
    elif mild and action == ["normal", "cut_power"]:
        return -1
    elif not danger and not mild and action != "normal":
        return -1
    else:
        return 0  # Nötr durumlarda her şey kabul

# Eğitim verisi olarak X ve Y listelerini hazırla ve kaydet


X = []
Y = []

for i in range(len(steps)):
    # Feature array: [cell_temp, current_draw, voltage, charging]
    X.append(np.array([
            temps[i],
            currents[i],
            voltages[i],
            int(chargings[i])
        ]).reshape(1, -1))
    # Modelin aksiyonu (Y)
    Y.append(actions[i])

# Eğitim verisini dosyaya kaydet
with open("train_data.pkl", "wb") as f:
    pickle.dump({"X": X, "Y": Y}, f)