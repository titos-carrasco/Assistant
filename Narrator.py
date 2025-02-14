import pyttsx3
import threading
import queue
import time


class Narrator:
    def __init__(self, volume=1, rate=160):
        self.engine = None
        self.queue = None
        self.running = False
        self.task = threading.Thread(target=self._speak, args=(volume, rate))

    def _speak(self, volume, rate):
        self.engine = pyttsx3.init()
        self.engine.setProperty("volume", volume)
        self.engine.setProperty("rate", rate)
        while self.running:
            try:
                text, event = self.queue.get_nowait()
                self.engine.say(text)
                self.engine.runAndWait()
                if event:
                    event.set()
            except queue.Empty:
                time.sleep(0.001)
        self.engine = None

    def start(self):
        self.queue = queue.Queue()
        self.running = True
        self.task.start()

    def stop(self):
        self.running = False
        self.task.join()
        self.queue = None

    def say(self, text, wait=False):
        if wait:
            event = threading.Event()
            self.queue.put_nowait([text, event])
            event.wait()
        else:
            self.queue.put_nowait([text, None])


# ---
if __name__ == "__main__":
    app = Narrator()
    app.start()
    data = [
        "Yo vivo en Granada, una ciudad pequeña que tiene monumentos muy importantes como la Alhambra.",
        "Aquí la comida es deliciosa y son famosos el gazpacho, el rebujito y el salmorejo.",
        "Mi nueva casa está en una calle ancha que tiene muchos árboles.",
    ]
    for text in data:
        print(text)
        app.say(text, wait=True)

    app.stop()
