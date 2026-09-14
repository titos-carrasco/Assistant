import time
import threading
import queue
import ollama
from Transcriber import Transcriber
from Narrator import Narrator

class Assistant:
    def __init__(
        self,
        llm,
        transcriber_model,
        speech_volume=1,
        speech_rate=160,
        fprompt="Contexto.txt",
    ):
        print("Inicializando LLM ...", flush=True)
        f = open(fprompt, "r", encoding="utf-8")
        prompt = ""
        prompt = prompt.join(f.readlines(-1))
        f.close()
        self.llm = llm
        self.messages = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        ollama.chat(
            model=self.llm,
            messages=self.messages,
            options={"num_predict": 1},
            stream=False,
        )
        self.queue = queue.Queue(0)

        self.transcriber = Transcriber(transcriber_model=transcriber_model)
        self.narrator = Narrator(speech_volume=speech_volume, speech_rate=speech_rate)

    def print_slow(self, text, delay=0.01):
        for letter in text:
            print(letter, end="", flush=True)
            time.sleep(delay)

    def _chat(self):
        resp = ollama.chat(
            model=self.llm,
            messages=self.messages,
            options={
                "num_ctx": 2048,
                "temperature": 0.5,
                "num_predict": 512,
            },
            stream=True,
        )

        for chunk in resp:
            text = chunk.message.content
            self.queue.put(text)
            time.sleep(0.001)
        self.queue.put(None)

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
                print(
                    "\r>>                                                  ",
                    end="",
                    flush=True,
                )
                print("\r>> Transcribiendo ...", end="", flush=True)
                transcription = self.transcribe_audio(audio)
                if transcription is None:
                    continue

                print(
                    "\r>>                                                  ",
                    end="",
                    flush=True,
                )
                print("\r>> ", end="", flush=True)
                print(transcription, flush=True)
                if transcription.lower() == "salir":
                    break

                # el audio transcrito lo enviamos al LLM y procesamos en un thread
                self.messages.append({"role": "user", "content": transcription})
                task = threading.Thread(target=self._chat)
                task.start()

                # presentamos lo que nos va devolviendo el LLM
                answer = ""
                word = ""
                dot = False
                while True:
                    text = self.queue.get()
                    if text is None:
                        break
                    answer = answer + text
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

                # agregamos la respuesta del llm
                self.messages.append({"role": "assistant", "content": answer})

        print("Eso es todo amigos !!!", flush=True)


# ---

# llm: phi3, llama3, deepseek-r1, deepseek-v2 -- https://ollama.com/search
# transcriber_model: tiny, base, small, medium large, large-v2
app = Assistant(llm="qwen2.5-coder", transcriber_model="small")
app.run()
