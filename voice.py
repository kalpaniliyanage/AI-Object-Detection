import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text):
    """Speak the given text."""
    engine.say(text)
    engine.runAndWait()
