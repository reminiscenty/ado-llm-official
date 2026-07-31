import time
from openai import OpenAI
from .utils import RateLimiter

REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

class GPT(object):
    def __init__(self,
                 model="4o-mini",
                 seed=114514,
                 max_token=2000,
                 temperature=0.5,
                 rate_limiter=None,
                 n_gen=1,
                 debug_mode=False
                 ):
        self.n_gen = n_gen
        self.model = model
        self.client = OpenAI()
        print(f"Using model: {self.model}")

        self.seed = seed
        if any(model.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES):
            max_token = max(max_token, 16000)
            temperature = 1
        self.max_token = max_token
        self.temperature = temperature
        self.systemMessage = "You are an analog circuit design expert that helps people find circuit sizing."

        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=40000, time_frame=60)
        else:
            self.rate_limiter = rate_limiter

        self.debug_mode = debug_mode

    def setSystemMessage(self, message:str):
        self.systemMessage = message

    def request(self, prompt):
        message = []
        # message.append({"role": "system", "content": "You are an AI assistant that helps people find information."})
        # message.append({"role": "system", "content": "You are an analog circuit design expert that helps people find circuit sizing."})
        message.append({"role": "system", "content":self.systemMessage})
        message.append({"role": "user", "content": prompt})

        MAX_RETRIES = 3

        response_raw = None
        max_completion_tokens = self.max_token
        for retry in range(MAX_RETRIES):
            try:
                start_time = time.time()
                self.rate_limiter.add_request(request_text=prompt, current_time=start_time)
                response_raw = self.client.chat.completions.create(
                    model=self.model,  # Use the model identifier for ChatGPT-3.5
                    messages=message,
                    seed=self.seed,
                    max_completion_tokens=max_completion_tokens,
                    temperature=self.temperature,
                    n=self.n_gen
                )
                if response_raw and response_raw.usage:
                    self.rate_limiter.add_request(request_token_count=response_raw.usage.total_tokens,
                                                  current_time=start_time)

                contents = [
                    choice.message.content
                    for choice in response_raw.choices
                ]
                if all(contents):
                    break

                finish_reason = response_raw.choices[0].finish_reason
                reasoning_tokens = None
                if response_raw.usage and response_raw.usage.completion_tokens_details:
                    reasoning_tokens = response_raw.usage.completion_tokens_details.reasoning_tokens
                print(
                    f'[AF] Empty LLM content on retry {retry + 1}/{MAX_RETRIES} '
                    f'(finish_reason={finish_reason}, max_completion_tokens={max_completion_tokens}, '
                    f'reasoning_tokens={reasoning_tokens}). Increasing token budget...'
                )
                max_completion_tokens = min(max_completion_tokens * 2, 32000)
            except Exception as e:
                print(f'[AF] RETRYING LLM REQUEST {retry + 1}/{MAX_RETRIES}...')
                print(response_raw)
                print(e)

        if response_raw is None:
            return None

        response = []
        for i in range(self.n_gen):
            # print(completion.choices[0].message)
            response.append(response_raw.choices[i].message.content or "")
        if self.debug_mode:
            for r in response:
                print(r)

        return response


