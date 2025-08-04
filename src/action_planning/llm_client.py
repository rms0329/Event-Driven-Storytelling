"""
OpenAI Compatible Interface for various providers.

* Supported providers and their API key env var
    - openai: OPENAI_API_KEY
    - groq, groq-native: GROQ_API_KEY
    - lepton: LEPTON_API_TOKEN
    - deepinfra: DEEPINFRA_API_TOKEN
    - anthropic: ANTHROPIC_API_KEY
    - cerebras: CEREBRAS_API_KEY
      (waitlisted for a free key)
    - gemini: GEMINI_API_KEY
"""

import os

from dotenv import load_dotenv


class NotSupportedProviderError(Exception):
    pass


class ModelRequiredInBaseUrlError(Exception):
    pass


load_dotenv()


class LLMClient:
    def __init__(self, provider="openai", model=None, api_key=None):
        """
        Some provider requires model for encoding base_url
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or self._get_api_key()
        self.base_url = self._get_base_url()
        self.client = self._get_client()

    def _get_api_key(self):
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "deepinfra":
            return os.getenv("DEEPINFRA_API_TOKEN")
        elif self.provider in ("groq-native", "groq"):
            return os.getenv("GROQ_API_KEY")
        elif self.provider == "lepton":
            return os.getenv("LEPTON_API_TOKEN")
        elif self.provider == "cerebras":
            return os.getenv("CEREBRAS_API_KEY")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "google":
            return os.getenv("GEMINI_API_KEY")

        raise NotSupportedProviderError(f"{self.provider} is not a supported provider.")

    def _get_base_url(self):
        if self.provider == "deepinfra":
            return "https://api.deepinfra.com/v1/openai"
        elif self.provider == "groq":
            return "https://api.groq.com/openai/v1"
        elif self.provider == "lepton":
            if not self.model:
                raise ModelRequiredInBaseUrlError(f"model is required for encoding base_url for {self.provider}.")
            return f"https://{self.model}.lepton.run/api/v1"
        elif self.provider == "cerebras":
            return "https://api.cerebras.ai/v1"
        elif self.provider in ("openai", "groq-native", "anthropic", "google"):
            return None

        raise NotSupportedProviderError(f"{self.provider} is not a supported provider.")

    def _get_client(self):
        """
        Use Native client for *-native providers.
        """
        if self.provider == "groq-native":
            import groq  # type: ignore[reportMissingImports]

            return groq.Groq(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic  # type: ignore[reportMissingImports]

            return anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "google":
            from google import genai  # type: ignore[reportMissingImports]

            return genai.Client(api_key=self.api_key)
        elif self.provider in ["openai", "deepinfra", "groq", "cerebras", "lepton"]:
            import openai  # type: ignore[reportMissingImports]

            return openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        raise NotSupportedProviderError(f"{self.provider} is not a supported provider.")

    def __getattr__(self, name):
        return getattr(self.client, name)

    def get_response(self, with_usage=False, **kwargs):
        if self.provider == "anthropic":
            return self._get_response_anthropic(with_usage, **kwargs)
        elif self.provider == "google":
            return self._get_response_google(with_usage, **kwargs)

        response = self.client.chat.completions.create(**kwargs)
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        response = response.choices[0].message.content
        if with_usage:
            return response, prompt_tokens, completion_tokens
        return response

    def _get_response_anthropic(self, with_usage=False, **kwargs):
        messages = kwargs.get("messages")
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = None

        response = self.client.messages.create(
            model=kwargs.get("model"),
            system=system_message,
            messages=messages,
            temperature=kwargs.get("temperature"),
            max_tokens=2048,
        )
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        response = response.content[0].text
        if with_usage:
            return response, prompt_tokens, completion_tokens
        return response

    def _get_response_google(self, with_usage=False, **kwargs):
        from google.genai.types import Content, GenerateContentConfig, Part  # type: ignore[reportMissingImports]

        messages = kwargs.get("messages")
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = None

        conversation_history = []
        for i, message in enumerate(messages[:-1]):  # the last message is the query
            if i % 2 == 0:
                assert message["role"] == "user"
            else:
                assert message["role"] == "assistant"

            conversation_history.append(
                Content(
                    parts=[Part(text=message["content"])],
                    role="user" if message["role"] == "user" else "model",
                )
            )

        response_format = kwargs.get("response_format", None)
        if response_format:
            assert response_format["type"] == "json_object"
            response_mime_type = "application/json"
        else:
            response_mime_type = None

        chat = self.client.chats.create(
            model=kwargs.get("model"),
            history=conversation_history,
            config=GenerateContentConfig(
                system_instruction=system_message,
                temperature=kwargs.get("temperature"),
                max_output_tokens=2048,
                response_mime_type=response_mime_type,
            ),
        )
        response = chat.send_message(messages[-1]["content"])

        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        response = response.text
        if with_usage:
            return response, prompt_tokens, completion_tokens
        return response


def main():
    provider = "openai"
    model = "gpt-4o-mini"
    client = LLMClient(provider=provider, model=model)
    args = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a simple calculator. Respond with only the result as the JSON format. Do not include any additional text or explanations.",
            },
            {"role": "user", "content": "1+2"},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    (
        response,
        prompt_tokens,
        completion_tokens,
    ) = client.get_response(with_usage=True, **args)
    print(prompt_tokens)
    print(completion_tokens)
    print(response)


if __name__ == "__main__":
    main()
