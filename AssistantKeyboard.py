import time
import threading
import queue
import ollama


class Assistant:
    def __init__(
        self,
        llm,
        fprompt="ContextAssistant.txt",
    ):
        print("Inicializando LLM ...", flush=True)
        f = open(fprompt, "r", encoding="utf-8")
        prompt = "".join(f.readlines(-1))
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

    def chat(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
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

        answer = ""
        for chunk in resp:
            text = chunk.message.content
            answer = answer + text
            self.queue.put(text)
            time.sleep(0.001)
        self.queue.put(None)
        self.messages.append({"role": "assistant", "content": answer})

    def print_slow(self, text, delay=0.01):
        for letter in text:
            print(letter, end="", flush=True)
            time.sleep(delay)

    def run(self):
        while True:
            prompt = input(f"{self.llm}>> ").strip()
            if prompt == "":
                continue
            if prompt.lower() == "salir":
                break

            # el prompt lo procesamoes en un thread para poder desplegarlo de manera apropiada
            task = threading.Thread(target=self.chat, args=(prompt,))
            task.start()

            # presentamos lo que nos va devolviendo el LLM
            word = ""
            dot = False
            while True:
                text = self.queue.get()
                if text is None:
                    break
                for ch in text:
                    if not dot:
                        word = word + ch
                        if ch in "\n:,;!?":
                            self.print_slow(word, delay=0.01)
                            word = ""
                        elif ch == ".":
                            dot = True
                    else:
                        dot = False
                        if ch in "0123456789":
                            word = word + ch
                        else:
                            self.print_slow(word, delay=0.01)
                            word = ch

            # queda un resto por mostrar
            if word != "":
                self.print_slow(word, delay=0.01)
            print(flush=True)

        print("Eso es todo amigos !!!", flush=True)


# ---

# llm: phi3, llama3, deepseek-r1, deepseek-v2 -- https://ollama.com/search
# transcriber_model: tiny, base, small, medium large, large-v2
app = Assistant(llm="qwen2.5-coder")
app.run()
