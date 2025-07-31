import hashlib
import random

import numpy as np

from refleksnetTrainDataGenerator import ReflexNetGenerator
from cache import ReflexCache
from visualizer import plot_simulation
from stateSimulator import StateSimulator
import pickle
from RefleksNet import ReflexNet

# Simülasyon verileri listeleri
steps = []
temps = []
currents = []
voltages = []
chargings = []
actions = []
cache_hits = []








# Aksiyonlar
ACTIONS = ["cut_power", "cool_down", "limit_current","increase_current", "normal"]

# Sistem bileşenleri
model = ReflexNetGenerator()
cache = ReflexCache()

# Simülasyon döngüsü
GREEN = '\033[92m'
RESET = '\033[0m'
simulator = StateSimulator()
action = "normal"  # Başlangıç aksiyonu
effect_timer = 0  # Aksiyon etkisi süresi

for step in range(101):
    state = simulator.step(action, effect_timer)
    key = state

    action =cache.find_similar_state_action(key)
    source = "CACHE"
    cache_hit = True
    if action is None:
        action = model.predict(state)
        effect_timer= model.action_timer
        if action != "normal":
            # Aksiyon normal değilse, cache'e kaydet
            cache.store(key, action)
        source = "RefleksNet"
        cache_hit = False



    # Verileri kaydet
    steps.append(step)
    temps.append(state["cell_temp"])
    currents.append(state["current_draw"])
    voltages.append(state["voltage"])
    chargings.append(state["charging"])
    actions.append(action)
    cache_hits.append(cache_hit)

    if source == "CACHE":
        print(f"{GREEN}[{step}] ({source}) State: {state} → Action: {action} {RESET}")
    else:
        print(f"[{step}] ({source}) State: {state} → Action: {action} ")
print(f"Stored Cache: {cache.len()}")
print(f"Cache Use: {cache_hits.count(True)}")
print(f"Eroors: {simulator.error_amount}")


