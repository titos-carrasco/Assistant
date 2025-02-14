import ollama
from Transcriber import Transcriber
from Narrator import Narrator
import time


class Assistant:
    def __init__(self):
        pass

    def run(self, model="llama3"):
        print("Inicializando Transcriptor...", flush=True)
        transcriber = Transcriber()
        transcriber.start()

        print("Inicializando Narrador ...", flush=True)
        narrator = Narrator()
        narrator.start()

        print("Inicializando LLM ...")
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": ""},
            ],
            options={"num_predict": 1},
            stream=False,
        )

        print("Inicialización finalizada. Diga 'Salir' para finalizar", flush=True)
        prompt = "\n>>"
        show_prompt = True
        while True:
            if show_prompt:
                print(prompt, end=" ", flush=True)
                show_prompt = False
            text = transcriber.getTranscription()
            if text is None:
                time.sleep(0.001)
            else:
                print(text, flush=True)
                if text == "Salir":
                    break

                resp = ollama.chat(
                    model="deepseek-r1:latest",
                    messages=[
                        {"role": "user", "content": text},
                    ],
                    options={"num_ctx": 2048, "temperature": 0.5, "num_predict": 100},
                    stream=True,
                )
                for chunk in resp:
                    for letter in chunk.message.content:
                        print(letter, end="", flush=True)
                        time.sleep(0.01)
                print()
                show_prompt = True

        print("Finalizando Narrador ...", flush=True)
        narrator.stop()

        print("Finalizando Transcriptor ...", flush=True)
        transcriber.stop()

        print("Eso es todo amigos !!!", flush=True)


# ---

app = Assistant()
app.run(model="deepseek-r1:latest")
