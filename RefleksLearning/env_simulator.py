class Environment:
    def __init__(self):
        self.vibration = 0
        self.stress = 0
        self.noise = 0

    def apply_action(self, action):
        """Dış etkiyi uygula"""
        if action["type"] == "LiftObject":
            weight = action.get("weight", 0)
            self.stress += weight * 0.8
            self.vibration += weight * 0.3
        elif action["type"] == "PowerSurge":
            intensity = action.get("intensity", 0)
            self.noise += intensity * 1.0
        elif action["type"] == "HumanNear":
            distance = action.get("distance", 1.0)
            self.vibration += max(0, 100 - distance * 30)
        elif action["type"] == "NormalOp":
            self.vibration = max(0, self.vibration - 5)
            self.stress = max(0, self.stress - 5)
            self.noise = max(0, self.noise - 5)

    def get_state(self):
        return {
            "vibration": self.vibration,
            "stress": self.stress,
            "noise": self.noise,
        }

    def reset(self):
        self.__init__()

    def stabilization(self):
        self.vibration = max(0, self.vibration - 20)
        self.stress = max(0, self.stress - 20)
        self.noise = max(0, self.noise - 20)

