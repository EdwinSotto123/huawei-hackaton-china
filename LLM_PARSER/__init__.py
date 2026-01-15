"""
LLM Parser Module
=================
Converts raw sign language predictions into natural, rich text using LLM.

Example:
    Input:  "YO EDWIN HOLA"
    Output: "¡Hola! Me llamo Edwin, mucho gusto."
"""

from .parser import SignToTextParser, parse_signs
from .deepseek_client import DeepSeekClient

__all__ = [
    "SignToTextParser",
    "DeepSeekClient", 
    "parse_signs"
]
