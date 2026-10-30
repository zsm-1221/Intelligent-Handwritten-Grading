"""Generic OCR-like corruptions, never learned from held-out handwriting."""
import random

CONFUSIONS = {"o":"0", "l":"1", "i":"l", "m":"rn", "n":"ri", "c":"e", "e":"c"}

def corrupt(text, rng, probability=0.035):
    result = []
    for char in text:
        if rng.random() < probability:
            mode = rng.random()
            if char.lower() in CONFUSIONS and mode < .6:
                char = CONFUSIONS[char.lower()]
            elif mode < .8:
                char = ""
            elif char == " ":
                char = ""
        result.append(char)
    return "".join(result)
