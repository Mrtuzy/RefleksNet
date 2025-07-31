# cache.py

class ReflexCache:
    def __init__(self):
        self.cache = []



    def store(self, key, action):
        self.cache.append({
            "state": key,
            "action": action
        })
    def get(self, key):
        """
        Verilen state için aksiyonu döndürür.
        Eğer state bulunamazsa None döner.
        """
        for entry in self.cache:
            if entry["state"] == key:
                return entry["action"]
        return None

    def len(self):
        return len(self.cache)

    def find_similar_state_action(self, state, threshold=5.0):
        """
        Benzer bir state bulur ve aksiyonunu döndürür.
        saved_states: [{ "state": {...}, "action": "cut_power" }, ...]
        threshold: Benzerlik için maksimum mesafe
        """
        min_dist = float("inf")
        best_action = None
        for entry in self.cache:
            s = entry["state"]
            # Öznitelik farklarının toplamı (L1 mesafesi)
            dist = (
                    abs(state["cell_temp"] - s["cell_temp"]) +
                    abs(state["current_draw"] - s["current_draw"]) +
                    abs(state["voltage"] - s["voltage"]) +
                    abs(int(state["charging"]) - int(s["charging"]))
            )
            if dist < min_dist and dist <= threshold:
                min_dist = dist
                best_action = entry["action"]
        return best_action  # None dönerse benzer yok