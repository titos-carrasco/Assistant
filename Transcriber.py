import speech_recognition as sr
import whisper
import numpy as np
import threading
import queue
import time


class Transcriber:
    def __init__(self, energy_threshold=2000, whisper_model="base"):
        # captura de audio
        self.source = sr.Microphone(sample_rate=16000)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.7

        self.capture_task = threading.Thread(target=self._capture)
        self.capture_queue = None
        self.capture_running = False

        # audio a texto
        self.transcribe_task = threading.Thread(target=self._transcribe)
        self.transcribe_queue = None
        self.transcribe_running = False

        self.whisper = whisper.load_model(whisper_model)

    def _capture(self):
        while self.capture_running:
            try:
                audio = self.recognizer.listen(self.source, timeout=1).get_wav_data()
            except sr.WaitTimeoutError:
                continue
            self.capture_queue.put_nowait(audio)

    def _transcribe(self):
        while self.transcribe_running:
            try:
                audio = self.capture_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
                continue
            arr = np.frombuffer(audio, dtype=np.int16)
            arr = arr.astype(np.float32) / 32768.0
            result = self.whisper.transcribe(audio=arr, fp16=False, language="Spanish")
            text = result["text"].strip()
            if text == "":
                continue
            self.transcribe_queue.put_nowait(text)

    # ---

    def start(self):
        self.source.__enter__()
        self.recognizer.adjust_for_ambient_noise(self.source, 2)

        self.transcribe_queue = queue.Queue()
        self.capture_queue = queue.Queue()

        self.transcribe_running = True
        self.transcribe_task.start()

        self.capture_running = True
        self.capture_task.start()

    def stop(self):
        self.capture_running = False
        self.capture_task.join()

        self.transcribe_running = False
        self.transcribe_task.join()

        self.transcribe_queue = None
        self.capture_queue = None

    def getTranscription(self):
        try:
            return self.transcribe_queue.get_nowait()
        except queue.Empty:
            return None


# ---
if __name__ == "__main__":
    app = Transcriber(2000, "base")
    app.start()
    while True:
        text = app.getTranscription()
        if text is None:
            time.sleep(0.001)
        else:
            print(text, flush=True)
            if text == "Salir":
                break
    app.stop()
