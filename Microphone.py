import whisper
import pyttsx3
import speech_recognition as sr
import numpy as np
import time


class Assistant:
    def __init__(self, transcriber_model, speech_volume=1, speech_rate=160):
        print("Inicializando Transcriptor...", flush=True)
        self.transcriber = whisper.load_model(transcriber_model)
        self.audio_recognizer = sr.Recognizer()
        self.audio_recognizer.energy_threshold = 2000
        self.audio_recognizer.dynamic_energy_threshold = False
        self.audio_recognizer.pause_threshold = 1.0

        print("Inicializando Narrador...", flush=True)
        self.speech_engine = pyttsx3.init()
        self.speech_engine.setProperty("volume", speech_volume)
        self.speech_engine.setProperty("rate", speech_rate)

    def say(self, text):
        self.speech_engine.say(text)
        self.speech_engine.runAndWait()

    def capture_audio(self, mic, timeout=1):
        try:
            return self.audio_recognizer.listen(mic, timeout=timeout).get_wav_data()
        except sr.WaitTimeoutError:
            return None

    def transcribe_audio(self, audio):
        arr = np.frombuffer(audio, dtype=np.int16)
        arr = arr.astype(np.float32) / 32768.0
        result = self.transcriber.transcribe(audio=arr, fp16=False, language="Spanish")
        transcription = result["text"].strip()
        if transcription == "":
            return None
        else:
            return transcription

    def run(self):
        print("Ajustando a ruido ambiental ...", flush=True)
        with sr.Microphone(sample_rate=16000) as mic:
            self.audio_recognizer.adjust_for_ambient_noise(mic, 2)

            print("Inicialización finalizada", flush=True)
            print("Diga 'Salir' para finalizar", flush=True)

            prompt = "\n>> Escuchando ..."
            show_prompt = True
            while True:
                # esperamos por un nuevo audio
                if show_prompt:
                    print(prompt, end=" ", flush=True)
                    show_prompt = False

                audio = self.capture_audio(mic, timeout=1)
                if audio is None:
                    continue

                # transcribimos el audio
                print("\r>> Transcribiendo ...  ", end="", flush=True)
                transcription = self.transcribe_audio(audio)
                if transcription is None:
                    continue

                print()
                print(transcription, end="", flush=True)
                if transcription.lower() == "salir":
                    break
                self.say(transcription)
                show_prompt = True


# ---

app = Assistant(transcriber_model="small")
app.run()
