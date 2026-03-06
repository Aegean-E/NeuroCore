"""
Structured Output Enforcement Module

Provides guaranteed schema-valid JSON output from LLM calls using:
1. Pydantic BaseModel schemas
2. Response format specification for JSON schema mode
3. Retry logic with validation error injection
"""

import asyncio
import logging
from typing import Type, Optional, List, Any
from pydantic import BaseModel, ValidationError

from core.llm import LLMBridge
from core.settings import settings

logger = logging.getLogger(__name__)


class StructuredOutputError(Exception):
    """
    Exception raised when structured output cannot be obtained
    after maximum retry attempts.
    """
    
    def __init__(self, message: str, schema: str = None, attempts: int = 0, last_error: str = None):
        self.schema = schema
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(message)
    
    def __str__(self):
        base = super().__str__()
        details = f" (Schema: {self.schema}, Attempts: {self.attempts}"
        if self.last_error:
            details += f", Last Error: {self.last_error}"
        return base + details + ")"


async def structured_completion(
    messages: List[dict],
    schema: Type[BaseModel],
    max_retries: int = 3,
    temperature: float = None,
    max_tokens: int = None,
    model: str = None,
    llm_bridge: LLMBridge = None,
    timeout: float = 60.0
) -> BaseModel:
    """
    LLM call that guarantees schema-valid output.
    
    This function:
    1. Uses the LLM's JSON schema response format
    2. Parses the response into the Pydantic model
    3. On validation failure, injects the error and retries
    
    Args:
        messages: Chat messages to send to the LLM
        schema: Pydantic BaseModel class to validate against
        max_retries: Maximum number of retry attempts (default: 3)
        temperature: Override default temperature
        max_tokens: Override default max tokens
        model: Override default model
        llm_bridge: Optional LLMBridge instance (creates new if not provided)
        timeout: Request timeout in seconds
        
    Returns:
        Parsed BaseModel instance
        
    Raises:
        StructuredOutputError: If all retry attempts fail
        
    Example:
        from pydantic import BaseModel
        from core.structured_output import structured_completion
        
        class Person(BaseModel):
            name: str
            age: int
            
        result = await structured_completion(
            messages=[{"role": "user", "content": "Create a person named John who is 30"}],
            schema=Person
        )
        print(result.name)  # "John"
        print(result.age)   # 30
    """
    # Create bridge if not provided
    if llm_bridge is None:
        llm_bridge = LLMBridge(
            base_url=settings.get("llm_api_url"),
            api_key=settings.get("llm_api_key"),
            timeout=timeout
        )
    
    schema_name = schema.__name__
    schema_json = schema.model_json_schema() if hasattr(schema, 'model_json_schema') else schema.schema()
    
    # Track original messages for retry context
    original_messages = messages.copy()
    
    for attempt in range(max_retries):
        try:
            # Prepare response format for JSON schema enforcement
            response_format = {
                "type": "json_object"  # Modern API format
            }
            
            # Add strict JSON schema if supported by the model
            # Note: This depends on the LLM provider support
            try:
                response_format["schema"] = schema_json
            except Exception as e:
                logger.debug(f"Could not add schema to response_format: {e}")
            
            # Make the LLM call
            response = await llm_bridge.chat_completion(
                messages=messages,
                model=model or settings.get("default_model"),
                temperature=temperature if temperature is not None else settings.get("temperature", 0.3),
                max_tokens=max_tokens or settings.get("max_tokens", 4096),
                response_format=response_format
            )
            
            # Check for API errors
            if "error" in response:
                error_msg = response.get("error", "Unknown error")
                logger.warning(f"LLM API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Add error context and retry
                messages.append({
                    "role": "user",
                    "content": f"API Error: {error_msg}. Please retry with valid JSON output."
                })
                continue
            
            # Extract content from response
            content = None
            try:
                choices = response.get("choices", [])
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
            except Exception as e:
                logger.warning(f"Failed to extract content from response: {e}")
                content = None
            
            if not content:
                logger.warning(f"Empty response (attempt {attempt + 1}/{max_retries})")
                messages.append({
                    "role": "user", 
                    "content": "Empty response. Please provide valid JSON output."
                })
                continue
            
            # Attempt to parse the content as the schema
            try:
                # First try parsing directly
                parsed = schema.model_validate_json(content) if hasattr(schema, 'model_validate_json') else schema.parse_raw(content)
                logger.debug(f"Successfully parsed structured output on attempt {attempt + 1}")
                return parsed
            except ValidationError as e:
                validation_errors = e.errors()
                error_summary = "; ".join([
                    f"{err.get('loc', 'unknown')}: {err.get('msg', 'invalid')}" 
                    for err in validation_errors
                ])
                
                logger.warning(f"Validation error (attempt {attempt + 1}/{max_retries}): {error_summary}")
                
                # Inject error feedback and retry
                messages.append({
                    "role": "assistant",
                    "content": content
                })
                messages.append({
                    "role": "user",
                    "content": f"Invalid format for {schema_name} schema: {error_summary}. Please fix and output valid JSON matching this schema: {schema_json}"
                })
                continue
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                messages.append({
                    "role": "user",
                    "content": "Request timed out. Please retry with a shorter response."
                })
            continue
            
        except Exception as e:
            logger.error(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                messages.append({
                    "role": "user",
                    "content": f"Error: {str(e)}. Please retry with valid JSON."
                })
            continue
    
    # All retries exhausted
    raise StructuredOutputError(
        f"Failed to get valid {schema_name} after {max_retries} attempts",
        schema=schema_name,
        attempts=max_retries
    )


async def structured_completion_with_fallback(
    messages: List[dict],
    schema: Type[BaseModel],
    max_retries: int = 3,
    fallback_value: Any = None,
    **kwargs
) -> Optional[BaseModel]:
    """
    Structured completion that returns fallback value on failure instead of raising.
    
    Args:
        messages: Chat messages
        schema: Pydantic BaseModel class
        max_retries: Maximum retry attempts
        fallback_value: Value to return on failure (or None)
        **kwargs: Additional args passed to structured_completion
        
    Returns:
        Parsed BaseModel instance or fallback_value
    """
    try:
        return await structured_completion(messages, schema, max_retries, **kwargs)
    except StructuredOutputError as e:
        logger.error(f"Structured output failed: {e}")
        return fallback_value


def create_json_schema(schema: Type[BaseModel]) -> dict:
    """
    Create a JSON schema dictionary from a Pydantic model.
    
    Args:
        schema: Pydantic BaseModel class
        
    Returns:
        JSON schema dictionary
    """
    if hasattr(schema, 'model_json_schema'):
        return schema.model_json_schema()
    return schema.schema()


# Convenience function for common schemas
async def structured_chat(
    messages: List[dict],
    response_schema: Type[BaseModel],
    system_prompt: str = None,
    **kwargs
) -> BaseModel:
    """
    Simplified interface for structured chat completions.
    
    Args:
        messages: User messages (system prompt will be prepended if provided)
        response_schema: Expected response schema
        system_prompt: Optional system prompt to prepend
        **kwargs: Additional args for structured_completion
        
    Returns:
        Parsed response
    """
    full_messages = []
    
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    
    full_messages.extend(messages)
    
    return await structured_completion(
        messages=full_messages,
        schema=response_schema,
        **kwargs
    )

