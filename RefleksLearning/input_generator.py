import random


def generate_random_input():
    choice = random.choice(["LiftObject", "PowerSurge", "HumanNear", "NormalOp"])

    if choice == "LiftObject":
        return {"type": "LiftObject", "weight": random.choice([5, 20, 50, 70])}
    elif choice == "PowerSurge":
        return {"type": "PowerSurge", "intensity": random.choice([10, 30, 80])}
    elif choice == "HumanNear":
        return {"type": "HumanNear", "distance": round(random.uniform(0.3, 2.0), 2)}
    else:
        return {"type": "NormalOp"}
