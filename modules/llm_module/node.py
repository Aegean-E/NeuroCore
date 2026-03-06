import asyncio
import logging
from core.llm import LLMBridge
from core.settings import settings

logger = logging.getLogger(__name__)

class ConfigLoader:
    @staticmethod
    def get_config():
        return settings.settings

# Default values for LLM parameters (used as explicit fallbacks)
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120.0  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds

class LLMExecutor:
    def __init__(self):
        self.bridge = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key")
        )

    async def receive(self, input_data: dict, config: dict = None) -> dict:
        """
        Executes LLM completion.
        Precedence: Node Config > Input Data > Module Defaults > Hardcoded Defaults
        
        Config options:
        - timeout (float, default 120): Timeout in seconds for the LLM call (0 = disabled)
        - max_retries (int, default 3): Number of retries on failure
        - retry_delay (float, default 1.0): Base delay in seconds for exponential backoff
        """
        # Bug fix: guard against None input
        if input_data is None:
            return {}

        config = config or {}

        # Get timeout, retry config from node config
        timeout = float(config.get("timeout", DEFAULT_TIMEOUT))
        max_retries = int(config.get("max_retries", DEFAULT_MAX_RETRIES))
        retry_delay = float(config.get("retry_delay", DEFAULT_RETRY_DELAY))

        module_defaults = ConfigLoader.get_config()

        # 1. Start with Module Defaults (with explicit fallbacks to hardcoded defaults)
        final_params = {
            "model": module_defaults.get("default_model", DEFAULT_MODEL),
            "temperature": module_defaults.get("temperature", DEFAULT_TEMPERATURE),
            "max_tokens": module_defaults.get("max_tokens", DEFAULT_MAX_TOKENS)
        }

        # 2. Override with Input Data (only specific keys)
        for key in ["model", "temperature", "max_tokens", "tools", "tool_choice"]:
            if key in input_data:
                final_params[key] = input_data[key]

        # 3. Override with Node Config
        for key in ["model", "temperature", "max_tokens", "tools", "tool_choice"]:
            if key in config:
                final_params[key] = config[key]

        messages = input_data.get("messages", [])
        
        # Execute with timeout and retry logic
        return await self._execute_with_retry_and_timeout(
            messages=messages,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **final_params
        )

    async def _execute_with_retry_and_timeout(self, messages: list, timeout: float, max_retries: int, retry_delay: float, **kwargs) -> dict:
        """
        Execute LLM call with optional timeout and retry logic.
        """
        async def _execute():
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return await self.bridge.chat_completion(
                        messages=messages,
                        **kwargs
                    )
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        # Exponential backoff: delay doubles with each retry
                        delay = retry_delay * (2 ** attempt)
                        logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
            
            # All retries exhausted
            logger.error(f"LLM call failed after {max_retries + 1} attempts: {last_error}")
            return {"error": str(last_error), "choices": []}

        # Execute with optional timeout
        try:
            if timeout > 0:
                return await asyncio.wait_for(_execute(), timeout=timeout)
            else:
                return await _execute()
        except asyncio.TimeoutError:
            logger.error(f"LLM call timed out after {timeout}s")
            return {"error": f"Timeout after {timeout}s", "choices": []}

    async def send(self, processed_data: dict) -> dict:
        return processed_data

async def get_executor_class(node_type_id: str):
    if node_type_id == 'llm_module':
        return LLMExecutor
    return None