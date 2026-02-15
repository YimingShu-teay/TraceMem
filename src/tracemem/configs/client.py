from openai import OpenAI
from typing import Optional
import logging
logger = logging.getLogger(__name__)
import time

class Client:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = ""):
        """
        Initialize LLM client
        
        Args:
            api_key: OpenAI API key
            model: Model name
            base_url: API base URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Configuration parameters
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30.0

        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        self._call_count = 0

    def client_response(self, system_prompt,input_prompt):
        for attempt in range(self.max_retries):
            try:            
                response = self.client.chat.completions.create(
                    model=self.model,  
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_prompt}
                    ],
                    max_tokens=16000,
                    temperature=0.1,
                    stream=False
                )
                # if hasattr(response, 'usage'):
                #     usage = response.usage
                #     self._total_usage["prompt_tokens"] += usage.prompt_tokens
                #     self._total_usage["completion_tokens"] += usage.completion_tokens
                #     self._total_usage["total_tokens"] += usage.total_tokens

                # self._call_count += 1
                
                result = response.choices[0].message.content
                return result
                
            except Exception as e:
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise e