def should_stop(state, thresholds):
    for key, value in thresholds.items():
        if value is not None and state[key] > value:
            return True
    return False
