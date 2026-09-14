import pyttsx3


class Narrator:
    def __init__(self, volume=1, rate=160):
        print("Inicializando Narrador...", flush=True)
        self.speech_engine = pyttsx3.init()
        self.speech_engine.setProperty("volume", volume)
        self.speech_engine.setProperty("rate", rate)

    def say(self, text):
        self.speech_engine.say(text)
        self.speech_engine.runAndWait()


# ---
if __name__ == "__main__":
    # nuestro narrador
    narrator = Narrator()

    # show time
    while True:
        # obtenemos el texto a verbalizar
        text = input("Ingrese el texto a verbalizar (o 'salir' para finalizar): ")
        if text.lower() == "salir":
            break

        # verbalizamos el texto
        narrator.say(text)
