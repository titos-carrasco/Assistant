import ollama
import whisper
import pyttsx3
import speech_recognition as sr
import numpy as np
import time


class Assistant:
    def __init__(self, llm, transcriber_model, speech_volume=1, speech_rate=160):
        print("Inicializando LLM ...", flush=True)
        ollama.chat(
            model=llm,
            messages=[
                {"role": "user", "content": ""},
            ],
            options={"num_predict": 1},
            stream=False,
        )

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

    def print_slow(self, text, delay=0.01):
        for letter in text:
            print(letter, end="", flush=True)
            time.sleep(delay)

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

            prompt = "\n>>"
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
                transcription = self.transcribe_audio(audio)
                if transcription is None:
                    continue

                print(transcription, flush=True)
                if transcription == "Salir":
                    break

                # el audio transcriot lo enviamos al LLM
                resp = ollama.chat(
                    model="deepseek-r1:latest",
                    messages=[
                        {
                            "role": "system",
                            "content": "Realiza todo tu razonamiento en español y responde únicamente en ese idioma.",
                        },
                        {"role": "user", "content": transcription},
                    ],
                    options={
                        "num_ctx": 2048,
                        "temperature": 0.5,
                        "num_predict": 512,
                    },
                    stream=True,
                )

                # presentamos lo que nos va devolviendo el LLM
                word = ""
                dot = False
                for chunk in resp:
                    text = chunk.message.content
                    for ch in text:
                        if not dot:
                            word = word + ch
                            if ch in "\n:,;!?":
                                self.print_slow(word, delay=0.01)
                                self.say(word)
                                word = ""
                            elif ch == ".":
                                dot = True
                        else:
                            dot = False
                            if ch in "0123456789":
                                word = word + ch
                            else:
                                self.print_slow(word, delay=0.01)
                                self.say(word)
                                word = ch

                # queda un resto por mostrar
                if word != "":
                    self.print_slow(word, delay=0.01)
                    self.say(word)

                print()
                show_prompt = True

        print("Eso es todo amigos !!!", flush=True)


# ---

app = Assistant("deepseek-r1:latest", "base")
app.run()
