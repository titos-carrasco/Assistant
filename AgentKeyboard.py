import time
import threading
import queue
import ollama
import json


class Assistant:
    def __init__(
        self,
        llm,
        fprompt="ContextAgent.txt",
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

    def chat(self):
        resp = ollama.chat(
            model=self.llm,
            messages=self.messages,
            options={
                "num_ctx": 2048,
                "temperature": 0.5,
                "num_predict": 512,
            },
            format = "json",
            # stream=True,
        )
        return resp

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

            self.messages.append({"role": "user", "content": prompt})
            resp = self.chat()
            json_string = resp.message.content
            self.messages.append({"role": "assistant", "content": json_string}) 

            data = json.loads(json_string)
            print(data, flush=True)
            # self.print_slow(word, delay=0.01)
            print(flush=True)

        print("Eso es todo amigos !!!", flush=True)


# ---

# llm: phi3, llama3, deepseek-r1, deepseek-v2 -- https://ollama.com/search
# transcriber_model: tiny, base, small, medium large, large-v2
app = Assistant(llm="qwen2.5-coder")
app.run()
