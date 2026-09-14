import whisper
import speech_recognition as sr
import numpy as np


class Transcriber:
    def __init__(self, transcriber_model):
        print("Inicializando Transcriptor...", flush=True)
        self.model = whisper.load_model(transcriber_model)
        self.audio_recognizer = sr.Recognizer()
        self.audio_recognizer.energy_threshold = 2000
        self.audio_recognizer.dynamic_energy_threshold = False
        self.audio_recognizer.pause_threshold = 1.0
        self.mic = None

    def noise_adjust(self):
        print("Ajustando a ruido ambiental ...", flush=True)
        self.audio_recognizer.adjust_for_ambient_noise(self.mic, 2)

    def capture_audio(self, timeout=1):
        try:
            return self.audio_recognizer.listen(
                self.mic, timeout=timeout
            ).get_wav_data()
        except sr.WaitTimeoutError:
            return None

    def transcribe_audio(self, audio):
        arr = np.frombuffer(audio, dtype=np.int16)
        arr = arr.astype(np.float32) / 32768.0
        result = self.model.transcribe(audio=arr, fp16=False, language="es")
        transcription = result["text"].strip()
        if transcription == "":
            return None
        else:
            return transcription


# ---
if __name__ == "__main__":
    # nuestro transcriptor
    transcriber = Transcriber(transcriber_model="small")

    # necesitamo el microfono para capturar el audio
    mic = sr.Microphone(sample_rate=16000)
    mic.__enter__()
    transcriber.mic = mic

    # show time
    try:
        # ajustamos el transcriptor al ruido ambiental
        transcriber.noise_adjust()

        # show time
        prompt = "\n>> Escuchando ..."
        show_prompt = True
        while True:

            # mostramos el prompt
            if show_prompt:
                print(prompt, end=" ", flush=True)
                show_prompt = False

            # capturamos un nuevo audio
            audio = transcriber.capture_audio()
            if audio is None:
                continue

            # transcribimos el audio
            print("\r>> Transcribiendo ...  ", end="", flush=True)
            transcription = transcriber.transcribe_audio(audio)
            if transcription is None:
                continue

            # mostramos la transcripcion
            print()
            print(transcription, end="", flush=True)
            if transcription.lower() == "salir":
                break
            show_prompt = True
    finally:
        mic.__exit__(None, None, None)
