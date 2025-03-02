import re
import json
import httpx
import numpy as np
import traceback  # Add this import for detailed error tracing
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Type, cast, TypeVar, Union

from pydantic import BaseModel

# Define our own token pattern instead of importing it
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Import the base classes from fast-graphrag
from fast_graphrag._llm._base import BaseLLMService, BaseEmbeddingService
from fast_graphrag._models import BaseModelAlias
from fast_graphrag._utils import logger

# Define T_model ourselves to match the usage in BaseLLMService
T_model = TypeVar("T_model", bound=Union[BaseModel, BaseModelAlias])

# Default Ollama endpoint
OLLAMA_URL = "http://localhost:11434"

###############################################################################
# Ollama LLM Service for local inference (e.g. mistral-small or other LLMs)
###############################################################################
@dataclass
class OllamaLLMService(BaseLLMService):
    """
    A local LLM client that sends prompts to an Ollama server at http://localhost:11434.
    By default, uses the 'mistral-small' model, but can be changed.
    """
    model: str = field(default="mistral-small")
    api_key: Optional[str] = field(default=None)  # Not used for Ollama, but required by BaseLLMService interface
    ollama_url: str = field(default=OLLAMA_URL)  # Make the URL configurable

    def __post_init__(self):
        # We don't have a specialized tokenizer, so we just do a naive token count
        self.encoding = None
        logger.info(f"Ollama LLM Service initialized with model: {self.model} at URL: {self.ollama_url}")

    def count_tokens(self, text: str) -> int:
        """
        Fallback to a naive token count using our local TOKEN_PATTERN.
        """
        return len(TOKEN_PATTERN.findall(text))

    async def send_message(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[T_model] | None = None,
        **kwargs: Any,
    ) -> Tuple[T_model, list[dict[str, str]]]:
        """
        Send a message to the local Ollama LLM server and get a structured (JSON) response.

        We do a blocking / single-shot call, ignoring streaming for simplicity. 
        If you want streaming, you'll need to adapt code to read from the streaming endpoint.
        """
        logger.info(f"OllamaLLMService.send_message called with model: {self.model}")
        
        # Combine system + user messages into one final prompt
        # Because some LLMs rely on role-based chat, we'll just flatten them
        prompt_text = ""
        if system_prompt:
            prompt_text += f"[SYSTEM MESSAGE]\n{system_prompt}\n\n"
            logger.debug(f"Added system prompt: {system_prompt[:100]}...")
        
        if history_messages:
            logger.debug(f"Processing {len(history_messages)} history messages")
            for msg in history_messages:
                if msg["role"] == "assistant":
                    prompt_text += f"[ASSISTANT MESSAGE]\n{msg['content']}\n\n"
                else:
                    prompt_text += f"[USER MESSAGE]\n{msg['content']}\n\n"
        
        # Now append the user's new prompt
        prompt_text += f"[USER MESSAGE]\n{prompt}"
        
        # Log token count for debugging
        token_count = self.count_tokens(prompt_text)
        logger.info(f"Total prompt token count (estimate): {token_count}")
        logger.debug(f"Ollama prompt being sent:\n{prompt_text[:500]}...\n")

        # Build the JSON payload for Ollama
        # See https://docs.ollama.ai/reference/generate for options
        temperature = kwargs.get("temperature", 0.1)
        top_p = kwargs.get("top_p", 1.0)
        
        payload = {
            "prompt": prompt_text,
            "model": self.model,
            # We'll include a simple set of parameters:
            "temperature": temperature,
            "top_p": top_p,
            # If you have a separate 'system' property, you can pass it, or just rely on the prompt
            # "system": system_prompt or "",
            "stop": ["\"\"\"", "```", "\n\n\n"],  # example stops
            "stream": False,  # for a single chunk
            "format": "json",  # return JSON
        }
        
        logger.info(f"Sending request to Ollama with temperature={temperature}, top_p={top_p}")

        async with httpx.AsyncClient(timeout=180) as client:
            try:
                # Log the full URL we're connecting to
                endpoint = f"{self.ollama_url}/api/generate"
                logger.info(f"Sending POST request to: {endpoint}")
                
                # Log a sample of the payload (not the full prompt to avoid log bloat)
                payload_log = payload.copy()
                if "prompt" in payload_log:
                    payload_log["prompt"] = payload_log["prompt"][:100] + "..." if len(payload_log["prompt"]) > 100 else payload_log["prompt"]
                logger.debug(f"Request payload (truncated): {payload_log}")
                
                # Make the request
                r = await client.post(endpoint, json=payload)
                
                # Log the response status
                logger.info(f"Ollama response status: {r.status_code}")
                
                # Check for HTTP errors
                r.raise_for_status()
                
                # Parse the response
                data = r.json()
                logger.debug(f"Ollama raw response data: {str(data)[:500]}...")
                
                # Extract the text response
                text_result = data.get("response", "")
                if not text_result:
                    logger.warning(f"Ollama returned empty response or missing 'response' field. Full data: {data}")
                
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
                try:
                    error_json = e.response.json()
                    logger.error(f"Ollama error details: {error_json}")
                except:
                    logger.error(f"Ollama error response (not JSON): {e.response.text}")
                raise
            except httpx.RequestError as e:
                logger.error(f"Ollama connection error: {str(e)}")
                logger.error(f"Connection details: {e}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Ollama response as JSON: {str(e)}")
                logger.error(f"Raw response text: {r.text[:1000]}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during Ollama request: {str(e)}")
                logger.error(f"Error traceback: {traceback.format_exc()}")
                raise

        # Sometimes local models produce partial JSON or extra text. We assume 
        # that for a structured model, we gave instructions to produce valid JSON.
        # If so, parse it carefully, or fallback to a string if it fails.
        final_response_str = text_result.strip()
        logger.debug(f"Ollama raw response text (first 500 chars):\n{final_response_str[:500]}...\n")
        
        # Try to clean up the response if it's not valid JSON but contains JSON
        if response_model:
            logger.info(f"Attempting to parse response as {response_model.__name__}")
            try:
                # Try to extract JSON if it's embedded in text
                json_start = final_response_str.find('{')
                json_end = final_response_str.rfind('}')
                
                if json_start >= 0 and json_end > json_start:
                    potential_json = final_response_str[json_start:json_end+1]
                    # Try to parse it
                    json_obj = json.loads(potential_json)
                    # If successful, use this as our response
                    final_response_str = json.dumps(json_obj)
                    logger.debug(f"Extracted and cleaned JSON: {final_response_str[:500]}...")
                else:
                    logger.warning(f"Could not find JSON markers in response")
            except Exception as e:
                logger.warning(f"Failed to extract JSON from response: {e}")
                logger.debug(f"JSON extraction attempted on: {final_response_str[:200]}...")

        # Return as the user expects. If we have a Pydantic model, parse it:
        if response_model:
            # If it's a fast_graphrag model that has a .Model inside it
            if issubclass(response_model, BaseModelAlias):
                # Try to parse JSON strictly
                try:
                    # parse raw JSON
                    logger.debug(f"Attempting to parse as BaseModelAlias: {response_model.__name__}")
                    pyd_model = response_model.Model.model_validate_json(final_response_str)
                    python_obj = cast(T_model, response_model.Model.to_dataclass(pyd_model))
                    # Return with the chat history
                    messages = [
                        {"role": "assistant", "content": final_response_str}
                    ]
                    logger.info(f"Successfully parsed response into {response_model.__name__}")
                    return python_obj, messages
                except Exception as e:
                    logger.warning(f"Failed to parse JSON into {response_model}: {e}")
                    logger.warning(f"JSON parsing error details: {traceback.format_exc()}")
                    # Return a fallback
                    messages = [
                        {"role": "assistant", "content": final_response_str}
                    ]
                    logger.info(f"Returning fallback string response")
                    return cast(T_model, final_response_str), messages
            else:
                # If it's a plain Pydantic model, parse it directly
                try:
                    logger.debug(f"Attempting to parse as regular Pydantic model: {response_model.__name__}")
                    pmodel = response_model.model_validate_json(final_response_str)
                    messages = [
                        {"role": "assistant", "content": final_response_str}
                    ]
                    logger.info(f"Successfully parsed response into {response_model.__name__}")
                    return pmodel, messages
                except Exception as e:
                    logger.warning(f"Failed to parse JSON into {response_model}: {e}")
                    logger.warning(f"JSON parsing error details: {traceback.format_exc()}")
                    # Return string fallback
                    messages = [
                        {"role": "assistant", "content": final_response_str}
                    ]
                    logger.info(f"Returning fallback string response")
                    return cast(T_model, final_response_str), messages
        else:
            # No model, just return text
            logger.info(f"No response model specified, returning raw text response")
            messages = [
                {"role": "assistant", "content": final_response_str}
            ]
            return cast(T_model, final_response_str), messages

###############################################################################
# Ollama Embedding Service
###############################################################################
@dataclass
class OllamaEmbeddingService(BaseEmbeddingService):
    """
    A local embedding client that calls Ollama's /embeddings endpoint
    with an embedding model like 'mxbai-embed-large'.
    """
    model: str = field(default="mxbai-embed-large")
    max_elements_per_request: int = 32
    api_key: Optional[str] = field(default=None)  # Not used for Ollama, but required by BaseEmbeddingService interface
    ollama_url: str = field(default=OLLAMA_URL)  # Make the URL configurable

    def __post_init__(self):
        logger.info(f"Ollama Embedding Service initialized with model: {self.model} at URL: {self.ollama_url}")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
        final_model = model or self.model
        logger.info(f"OllamaEmbeddingService.encode called with {len(texts)} texts using model: {final_model}")
        
        embeddings = []

        async with httpx.AsyncClient(timeout=180) as client:
            # We will batch the requests if we have more than max_elements_per_request
            for i in range(0, len(texts), self.max_elements_per_request):
                batch = texts[i : i + self.max_elements_per_request]
                logger.info(f"Processing embedding batch {i//self.max_elements_per_request + 1} of {(len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request}")
                
                # Log a sample of the first text in the batch
                sample_text = batch[0][:100] + "..." if len(batch[0]) > 100 else batch[0]
                logger.debug(f"Sample text from batch: {sample_text}")
                
                payload = {
                    "model": final_model,
                    "prompt": batch[0] if len(batch) == 1 else batch
                }
                
                endpoint = f"{self.ollama_url}/api/embed"
                logger.info(f"Sending embedding request to: {endpoint} with batch size={len(batch)}")
                
                try:
                    r = await client.post(endpoint, json=payload)
                    
                    # Log response status
                    logger.info(f"Ollama embedding response status: {r.status_code}")
                    
                    # Check for HTTP errors
                    r.raise_for_status()
                    
                    # Parse the response
                    data = r.json()
                    
                    if len(batch) == 1:
                        # Single embedding
                        embed_vec = data.get("embedding", [])
                        if not embed_vec:
                            logger.warning(f"Ollama returned empty embedding or missing 'embedding' field")
                        else:
                            logger.debug(f"Received single embedding with dimension: {len(embed_vec)}")
                        embeddings.append(embed_vec)
                    else:
                        # Multiple embeddings
                        batch_embeds = data.get("embeddings", [])
                        if not batch_embeds:
                            logger.warning(f"Ollama returned empty embeddings or missing 'embeddings' field")
                        else:
                            logger.debug(f"Received {len(batch_embeds)} embeddings with dimension: {len(batch_embeds[0]) if batch_embeds else 'unknown'}")
                        embeddings.extend(batch_embeds)
                        
                except httpx.HTTPStatusError as e:
                    logger.error(f"Ollama embedding HTTP error: {e.response.status_code} - {e.response.text}")
                    try:
                        error_json = e.response.json()
                        logger.error(f"Ollama embedding error details: {error_json}")
                    except:
                        logger.error(f"Ollama embedding error response (not JSON): {e.response.text}")
                    raise
                except httpx.RequestError as e:
                    logger.error(f"Ollama embedding connection error: {str(e)}")
                    logger.error(f"Connection details: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error during Ollama embedding request: {str(e)}")
                    logger.error(f"Error traceback: {traceback.format_exc()}")
                    raise

        # Convert to numpy
        if not embeddings:
            logger.error("No embeddings were generated! Returning empty array.")
            return np.array([], dtype=np.float32)
            
        arr = np.array(embeddings, dtype=np.float32)
        logger.info(f"Ollama embedding complete. Response shape: {arr.shape}")
        return arr 