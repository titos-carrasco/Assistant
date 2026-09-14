# Assistant

Asistente/Agente para interactuar con un LLM

* Transcriber.py: Clase para convertir audio a texto
* Narrator.py: Clase para convertir texto a audio
* Assistant.py: Asistente simple de  pregunta/respuesta activado por voz para interactuar conn un LLM bajo Ollama
* Context.txt: Contexto inicial entregado al LLM

COnfig.json

```json
{
    "configurations": [
        {
            "name": "Grok-qwen3-32b",
            "endpoint": "https://api.groq.com/openai/v1/chat/completions",
            "model": "qwen/qwen3-32b",
            "api_key": "xxx"
        },
        {
            "name": "DeekSeek",
            "endpoint": "",
            "model": "",
            "api_key": "xxx"
        }
    ]
}
```
