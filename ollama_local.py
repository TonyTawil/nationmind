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
OLLAMA_URL = "http://144.76.202.118:11434"

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
    debug_mode: bool = field(default=False)  # Enable extra verbose debugging

    def __post_init__(self):
        # We don't have a specialized tokenizer, so we just do a naive token count
        self.encoding = None
        logger.info(f"Ollama LLM Service initialized with model: {self.model} at URL: {self.ollama_url}")
        if self.debug_mode:
            logger.info("DEBUG MODE ENABLED: Will print verbose debugging information")

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
        temperature = kwargs.get("temperature", 0.1)
        top_p = kwargs.get("top_p", 1.0)
        
        # Check if we're using a structured model response
        format_schema = None
        if response_model:
            logger.info(f"Creating format schema for response model: {response_model.__name__}")
            try:
                # If it's a TGraph model, we need to create a specific schema
                if response_model.__name__ == "TGraph":
                    format_schema = {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {"type": "string"},
                                        "desc": {"type": "string"}
                                    },
                                    "required": ["name", "type", "desc"]
                                }
                            },
                            "relationships": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "target": {"type": "string"},
                                        "desc": {"type": "string"}
                                    },
                                    "required": ["source", "target", "desc"]
                                }
                            },
                            "other_relationships": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "target": {"type": "string"},
                                        "desc": {"type": "string"}
                                    },
                                    "required": ["source", "target", "desc"]
                                }
                            }
                        },
                        "required": ["entities", "relationships", "other_relationships"]
                    }
                    logger.debug(f"Created TGraph schema: {json.dumps(format_schema)[:200]}...")
            except Exception as e:
                logger.warning(f"Failed to create format schema: {e}")
                logger.warning(f"Schema creation error details: {traceback.format_exc()}")
                format_schema = None
        
        # Log the full URL we're connecting to
        if format_schema:
            # Use the chat endpoint for structured format
            endpoint = f"{self.ollama_url}/api/chat"
            
            # Reformat payload for chat endpoint
            chat_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "stream": False,
                "format": format_schema,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            logger.info(f"Using chat endpoint for structured format: {endpoint}")
            payload = chat_payload
        else:
            # Use generate endpoint for standard format
            endpoint = f"{self.ollama_url}/api/generate"
            logger.info(f"Using generate endpoint for standard format: {endpoint}")
            
            # Define the payload for the generate endpoint
            payload = {
                "model": self.model,
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
        
        logger.info(f"Sending POST request to: {endpoint}")

        async with httpx.AsyncClient(timeout=180) as client:
            try:
                # Log a sample of the payload (not the full prompt to avoid log bloat)
                payload_log = payload.copy()
                if "prompt" in payload_log:
                    payload_log["prompt"] = payload_log["prompt"][:100] + "..." if len(payload_log["prompt"]) > 100 else payload_log["prompt"]
                if "format" in payload_log and isinstance(payload_log["format"], dict):
                    payload_log["format"] = "structured schema (object)"
                logger.debug(f"Request payload (truncated): {payload_log}")
                
                # Make the request
                r = await client.post(endpoint, json=payload)
                
                # Log the response status
                logger.info(f"Ollama response status: {r.status_code}")
                
                # In debug mode, log the complete raw response text
                if self.debug_mode:
                    logger.info("========== COMPLETE RAW RESPONSE TEXT ==========")
                    logger.info(f"{r.text}")
                    logger.info("===============================================")
                
                # Check for HTTP errors
                r.raise_for_status()
                
                # Parse the response
                data = r.json()
                
                # Log the complete raw response for debugging
                logger.info("========== COMPLETE RAW OLLAMA RESPONSE ==========")
                logger.info(f"{json.dumps(data, indent=2)}")
                logger.info("==================================================")
                
                logger.debug(f"Ollama raw response data: {str(data)[:500]}...")
                
                # Extract the text response
                if format_schema:
                    # For structured format, the response might be in a different field
                    # Ollama sometimes returns metadata like model, total_duration, etc.
                    logger.debug(f"Raw response with format schema: {json.dumps(data)[:500]}...")
                    
                    # Check if we have a message field that contains our structured data (chat endpoint)
                    if "message" in data and "content" in data["message"]:
                        message_content = data["message"]["content"]
                        logger.info(f"Found message.content in response: {message_content[:200]}...")
                        try:
                            # Try to parse the message.content field as JSON
                            response_json = json.loads(message_content)
                            text_result = json.dumps(response_json)
                            logger.info(f"Successfully parsed JSON from message.content")
                        except json.JSONDecodeError as e:
                            logger.warning(f"message.content is not valid JSON: {e}")
                            text_result = message_content
                    # Check if we have a response field that contains our structured data (generate endpoint)
                    elif "response" in data:
                        try:
                            # Try to parse the response field as JSON
                            response_json = json.loads(data["response"])
                            text_result = json.dumps(response_json)
                            logger.debug(f"Extracted JSON from response field: {text_result[:500]}...")
                        except json.JSONDecodeError:
                            # If it's not valid JSON, use it as is
                            text_result = data["response"]
                            logger.warning(f"Response field is not valid JSON: {text_result[:200]}...")
                    else:
                        # If there's no response field, check if the data itself has our expected fields
                        if any(key in data for key in ["entities", "relationships", "other_relationships"]):
                            # This looks like our structured data
                            text_result = json.dumps(data)
                            logger.debug(f"Using full response as structured data: {text_result[:500]}...")
                        else:
                            # Last resort: create a minimal valid structure
                            logger.warning("Creating minimal valid structure as fallback")
                            fallback_data = {
                                "entities": [],
                                "relationships": [],
                                "other_relationships": []
                            }
                            text_result = json.dumps(fallback_data)
                else:
                    # For standard format, extract from the 'response' field
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
        if response_model and not format_schema:  # Only needed for non-structured format
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
                    
                    # Special handling for TGraph to ensure it has all required fields
                    if response_model.__name__ == "TGraph":
                        # Try to parse the JSON and add missing fields if needed
                        try:
                            json_obj = json.loads(final_response_str)
                            
                            # Log the complete parsed JSON object
                            logger.info("========== PARSED JSON OBJECT ==========")
                            logger.info(f"{json.dumps(json_obj, indent=2)}")
                            logger.info("=======================================")
                            
                            # Check if we got Ollama metadata instead of structured data
                            if "model" in json_obj and "message" not in json_obj and not any(key in json_obj for key in ["entities", "relationships", "other_relationships"]):
                                logger.warning("Received Ollama metadata instead of structured data, creating empty structure")
                                json_obj = {
                                    "entities": [],
                                    "relationships": [],
                                    "other_relationships": []
                                }
                            
                            # Check if all required fields exist
                            missing_fields = []
                            if "entities" not in json_obj:
                                missing_fields.append("entities")
                            if "relationships" not in json_obj:
                                missing_fields.append("relationships")
                            if "other_relationships" not in json_obj:
                                missing_fields.append("other_relationships")
                            
                            # Only modify if fields are missing
                            if missing_fields:
                                logger.warning(f"Missing required fields in JSON: {missing_fields}")
                                for field in missing_fields:
                                    logger.warning(f"Adding empty '{field}' field to JSON")
                                    json_obj[field] = []
                                
                                # Update the response string
                                final_response_str = json.dumps(json_obj)
                                logger.debug(f"Fixed JSON structure: {final_response_str[:200]}...")
                            else:
                                logger.info("JSON structure is valid, no modifications needed")
                        except Exception as e:
                            logger.warning(f"Failed to fix JSON structure: {e}")
                            logger.warning(f"Creating minimal valid TGraph structure as fallback")
                            final_response_str = json.dumps({
                                "entities": [],
                                "relationships": [],
                                "other_relationships": []
                            })
                    
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
                    
                    # Special handling for TAnswer model which needs to be wrapped
                    if response_model.__name__ == "TAnswer":
                        # Create a valid TAnswer JSON structure
                        answer_json = json.dumps({"answer": final_response_str})
                        logger.info(f"Wrapping response in TAnswer structure: {answer_json[:100]}...")
                        pmodel = response_model.model_validate_json(answer_json)
                    else:
                        pmodel = response_model.model_validate_json(final_response_str)
                    
                    messages = [
                        {"role": "assistant", "content": final_response_str}
                    ]
                    logger.info(f"Successfully parsed response into {response_model.__name__}")
                    return pmodel, messages
                except Exception as e:
                    logger.warning(f"Failed to parse JSON into {response_model}: {e}")
                    logger.warning(f"JSON parsing error details: {traceback.format_exc()}")
                    
                    # For TAnswer, create a valid object directly instead of returning a string
                    if response_model.__name__ == "TAnswer":
                        try:
                            # Create TAnswer directly
                            logger.info("Creating TAnswer object directly")
                            from fast_graphrag._models import TAnswer
                            answer_obj = TAnswer(answer=final_response_str)
                            messages = [
                                {"role": "assistant", "content": final_response_str}
                            ]
                            return answer_obj, messages
                        except Exception as inner_e:
                            logger.warning(f"Failed to create fallback TAnswer: {inner_e}")
                    
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
    debug_mode: bool = field(default=False)  # Add debug mode here too

    def __post_init__(self):
        logger.info(f"Ollama Embedding Service initialized with model: {self.model} at URL: {self.ollama_url}")
        if self.debug_mode:
            logger.info("DEBUG MODE ENABLED: Will print verbose embedding debugging information")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
        final_model = model or self.model
        logger.info(f"OllamaEmbeddingService.encode called with {len(texts)} texts using model: {final_model}")
        
        if not texts:
            logger.warning("Empty texts list provided to encode method")
            return np.array([], dtype=np.float32)
            
        embeddings = []

        async with httpx.AsyncClient(timeout=180) as client:
            # We will batch the requests if we have more than max_elements_per_request
            for i in range(0, len(texts), self.max_elements_per_request):
                batch = texts[i : i + self.max_elements_per_request]
                logger.info(f"Processing embedding batch {i//self.max_elements_per_request + 1} of {(len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request}")
                
                # Log a sample of the first text in the batch
                sample_text = batch[0][:100] + "..." if len(batch[0]) > 100 else batch[0]
                logger.debug(f"Sample text from batch: {sample_text}")
                
                # For Ollama, we need to handle single vs multiple texts differently
                if len(batch) == 1:
                    payload = {
                        "model": final_model,
                        "prompt": batch[0]
                    }
                else:
                    payload = {
                        "model": final_model,
                        "prompt": batch
                    }
                
                endpoint = f"{self.ollama_url}/api/embed"
                logger.info(f"Sending embedding request to: {endpoint} with batch size={len(batch)}")
                
                try:
                    r = await client.post(endpoint, json=payload)
                    
                    # Log response status
                    logger.info(f"Ollama embedding response status: {r.status_code}")
                    
                    # In debug mode, log the complete raw response
                    if self.debug_mode:
                        logger.info("========== COMPLETE EMBEDDING RESPONSE ==========")
                        logger.info(f"{r.text}")
                        logger.info("================================================")
                    
                    # Check for HTTP errors
                    r.raise_for_status()
                    
                    # Parse the response
                    data = r.json()
                    logger.info(f"Embedding response keys: {list(data.keys())}")
                    
                    # Log the complete response for debugging
                    logger.info("========== EMBEDDING RESPONSE DATA ==========")
                    logger.info(f"{json.dumps(data, indent=2)}")  # Log the full response, not just the first 500 chars
                    logger.info("===========================================")
                    
                    # Log additional details about the response structure
                    for key, value in data.items():
                        if isinstance(value, list):
                            logger.info(f"List field '{key}' has {len(value)} elements")
                            if value and len(value) > 0:
                                sample = value[0]
                                if isinstance(sample, list):
                                    logger.info(f"  First element is a list with {len(sample)} items")
                                    logger.info(f"  Sample values: {sample[:5]}...")
                                else:
                                    logger.info(f"  First element type: {type(sample).__name__}")
                        elif isinstance(value, dict):
                            logger.info(f"Dict field '{key}' has {len(value)} keys: {list(value.keys())}")
                        else:
                            logger.info(f"Field '{key}' has type {type(value).__name__}")
                    
                    if len(batch) == 1:
                        # Single embedding - check different possible response formats
                        if "embedding" in data:
                            embed_vec = data["embedding"]
                            logger.info(f"Found single embedding with dimension: {len(embed_vec)}")
                            embeddings.append(embed_vec)
                        elif "embeddings" in data and len(data["embeddings"]) > 0:
                            # Some Ollama versions might return a list even for single inputs
                            embed_vec = data["embeddings"][0]
                            logger.info(f"Found single embedding in 'embeddings' list with dimension: {len(embed_vec)}")
                            embeddings.append(embed_vec)
                        else:
                            # Try to find any array in the response that looks like an embedding
                            for key, value in data.items():
                                if isinstance(value, list) and len(value) > 0 and all(isinstance(x, (int, float)) for x in value[:10]):
                                    logger.warning(f"Using '{key}' field as embedding with dimension: {len(value)}")
                                    embeddings.append(value)
                                    break
                            else:
                                logger.error(f"Could not find embedding in response: {data}")
                                # Create a zero embedding as fallback (better than crashing)
                                logger.warning("Creating zero embedding as fallback")
                                embeddings.append([0.0] * 1536)  # Typical embedding dimension
                    else:
                        # Multiple embeddings
                        if "embeddings" in data:
                            batch_embeds = data["embeddings"]
                            if batch_embeds and len(batch_embeds) > 0:
                                logger.info(f"Found {len(batch_embeds)} embeddings with dimension: {len(batch_embeds[0])}")
                                embeddings.extend(batch_embeds)
                            else:
                                logger.error("Empty embeddings array in response")
                                # Create zero embeddings as fallback
                                logger.warning(f"Creating {len(batch)} zero embeddings as fallback")
                                for _ in range(len(batch)):
                                    embeddings.append([0.0] * 1536)  # Typical embedding dimension
                        else:
                            logger.error(f"Missing 'embeddings' field in response for batch request: {list(data.keys())}")
                            # Create zero embeddings as fallback
                            logger.warning(f"Creating {len(batch)} zero embeddings as fallback")
                            for _ in range(len(batch)):
                                embeddings.append([0.0] * 1536)  # Typical embedding dimension
                        
                except httpx.HTTPStatusError as e:
                    logger.error(f"Ollama embedding HTTP error: {e.response.status_code} - {e.response.text}")
                    try:
                        error_json = e.response.json()
                        logger.error(f"Ollama embedding error details: {error_json}")
                    except:
                        logger.error(f"Ollama embedding error response (not JSON): {e.response.text}")
                    # Create fallback embeddings instead of failing
                    logger.warning(f"Creating {len(batch)} zero embeddings as fallback after HTTP error")
                    for _ in range(len(batch)):
                        embeddings.append([0.0] * 1536)  # Typical embedding dimension
                except httpx.RequestError as e:
                    logger.error(f"Ollama embedding connection error: {str(e)}")
                    logger.error(f"Connection details: {e}")
                    # Create fallback embeddings instead of failing
                    logger.warning(f"Creating {len(batch)} zero embeddings as fallback after connection error")
                    for _ in range(len(batch)):
                        embeddings.append([0.0] * 1536)  # Typical embedding dimension
                except Exception as e:
                    logger.error(f"Unexpected error during Ollama embedding request: {str(e)}")
                    logger.error(f"Error traceback: {traceback.format_exc()}")
                    # Create fallback embeddings instead of failing
                    logger.warning(f"Creating {len(batch)} zero embeddings as fallback after unexpected error")
                    for _ in range(len(batch)):
                        embeddings.append([0.0] * 1536)  # Typical embedding dimension

        # Verify we have the right number of embeddings
        if len(embeddings) != len(texts):
            logger.error(f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}")
            # Pad with zeros if needed
            while len(embeddings) < len(texts):
                logger.warning(f"Adding zero embedding to match expected count")
                embeddings.append([0.0] * 1536)  # Typical embedding dimension
            # Truncate if we somehow got too many
            if len(embeddings) > len(texts):
                logger.warning(f"Truncating embeddings from {len(embeddings)} to {len(texts)}")
                embeddings = embeddings[:len(texts)]
            
        # Convert to numpy
        arr = np.array(embeddings, dtype=np.float32)
        logger.info(f"Ollama embedding complete. Response shape: {arr.shape}")
        return arr 