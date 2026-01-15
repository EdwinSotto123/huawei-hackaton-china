"""
DeepSeek API Client
====================
Client for interacting with DeepSeek LLM API using OpenAI SDK.

Installation:
    pip install openai
"""

import os
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI SDK required. Install with: pip install openai"
    )


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek API."""
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 2048
    temperature: float = 0.7


class DeepSeekClient:
    """
    Client for DeepSeek LLM API using OpenAI SDK.
    
    Usage:
        client = DeepSeekClient(api_key="your-api-key")
        response = client.chat("YO EDWIN HOLA")
        
        # Or with environment variable
        # export DEEPSEEK_API_KEY="your-api-key"
        client = DeepSeekClient()
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        max_tokens: int = 256,
        temperature: float = 0.7
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            base_url: API base URL
            model: Model to use (deepseek-chat, deepseek-coder, etc.)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 - 1.0)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. "
                "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )
        
        self.config = DeepSeekConfig(
            api_key=self.api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Initialize OpenAI client with DeepSeek endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Send a chat message and get response.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            history: Optional conversation history
            
        Returns:
            Model response text
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add history
        if history:
            messages.extend(history)
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Make API request using OpenAI SDK
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=False
        )
        
        return response.choices[0].message.content
    
    def chat_stream(
        self,
        message: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream chat response (generator).
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Yields:
            Response text chunks
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": message})
        
        # Stream response using OpenAI SDK
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


if __name__ == "__main__":
    # Test the client
    print("=" * 50)
    print("DeepSeek Client Test (OpenAI SDK)")
    print("=" * 50)
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if api_key:
        print("\n✅ API key found. Testing...\n")
        
        client = DeepSeekClient()
        
        # Test simple chat
        print("Test 1: Simple chat")
        response = client.chat("Say hello in Spanish")
        print(f"  Response: {response}\n")
        
        # Test with system prompt
        print("Test 2: With system prompt")
        response = client.chat(
            "YO NOMBRE EDWIN",
            system_prompt="Convert these sign language words to natural Spanish. Reply only with the sentence."
        )
        print(f"  Response: {response}\n")
        
        # Test streaming
        print("Test 3: Streaming")
        print("  Response: ", end="")
        for chunk in client.chat_stream("Count from 1 to 5"):
            print(chunk, end="", flush=True)
        print("\n")
        
        print("✅ All tests passed!")
    else:
        print("\n⚠️  DEEPSEEK_API_KEY not set.")
        print("Set it with: export DEEPSEEK_API_KEY='your-key'")
        print("Get your key at: https://platform.deepseek.com/")
    
    print("\n" + "=" * 50)
