import requests
import json
import re


class MyAssistant:
    def __init__(self, configuration):
        # el modelo a utilizar
        f = open("config.json", "r", encoding="utf-8")
        configs = json.load(f)
        f.close()
        for config in configs["configurations"]:
            if config["name"] == configuration:
                self.config = config
                break

        # el system prompt
        f = open("SystemPrompt.txt", "r", encoding="utf-8")
        system_prompt = f.read()
        f.close()

        # el historial de mensajes empieza con el con el system prompt
        self.messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

    def send(self):
        headers = {
            "Authorization": f"Bearer {self.config["api_key"]}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config["model"],
            "messages": self.messages,
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        try:
            response = requests.post(
                self.config["endpoint"], headers=headers, json=payload
            )
            if response.status_code != 200:
                print(
                    f"Error: {response.status_code} - {response.json().get('error', {}).get('message', 'Unknown error')}"
                )
                return ""
            return (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception as e:
            print(f"Error al enviar el mensaje: {e}")
            return ""

    def run(self):
        while True:
            # la entrada del usuario
            user_input = input("$: ")
            user_input = user_input.strip()
            if not user_input:
                continue

            # salimos al recibir esto
            if user_input.lower() in ["exit", "quit"]:
                print("Saliendo...")
                break

            # agregamos el mensaje del usuario al historial
            self.messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            # enviamos el historial al LLM
            response = self.send()
            if response:
                print(response)
                # agregamos la respuesta del LLM al historial
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )
                continue

                # la respuesta la estructuramos en una clase llamada Answer
                answer = type("Answer", (object,), {})
                answer.think = None
                answer.acciones=[]

                try:
                    answer.think = re.search(r'<think>(.*?)</think>', response, re.DOTALL).group(1).strip()
                except AttributeError:
                    pass
                
                for accion in re.findall(r'<json>(.*?)</json>', response, re.DOTALL):
                    try:
                        answer.acciones.append(json.loads(accion.strip()))
                    except json.JSONDecodeError:
                        pass

                # mostramos todo
                #print("Razonamiento: ", answer.think)
                for accion in answer.acciones:
                    print("Acción: ", accion)

# ---
app = MyAssistant("Grok-qwen3-32b")
app.run()
