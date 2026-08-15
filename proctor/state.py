"""Small shared runtime state used by the legacy detection loop."""

from collections import deque

me = ""
noise = 0
stop_detection = False
cheating_scores = deque(maxlen=None)
