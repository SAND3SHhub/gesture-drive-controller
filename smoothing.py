from collections import deque, Counter


class GestureSmoother:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)

    def update(self, gesture):
        self.history.append(gesture)

        counts = Counter(self.history)

        return counts.most_common(1)[0][0]