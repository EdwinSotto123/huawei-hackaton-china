"""
Sign to Text Parser
====================
Converts raw sign language predictions into natural, rich text using LLM.

Examples:
    "YO EDWIN"           → "Me llamo Edwin"
    "HOLA COMO ESTAR TU" → "¡Hola! ¿Cómo estás?"
    "GRACIAS AYUDA"      → "Gracias por tu ayuda"
"""

import os
import re
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum

from .deepseek_client import DeepSeekClient


class OutputStyle(Enum):
    """Output text style options."""
    CASUAL = "casual"           # Informal, friendly
    FORMAL = "formal"           # Professional, polite
    EXPRESSIVE = "expressive"   # With emotions and emphasis
    MINIMAL = "minimal"         # Short and direct


@dataclass
class ParserConfig:
    """Configuration for the Sign to Text Parser."""
    language: str = "es"  # Target language (es, en, etc.)
    style: OutputStyle = OutputStyle.CASUAL
    context: str = ""     # Additional context for better parsing
    include_punctuation: bool = True
    include_emotions: bool = True
    max_output_length: int = 200


# System prompts for different languages and styles
SYSTEM_PROMPTS = {
    "es": {
        OutputStyle.CASUAL: """Eres un asistente que convierte señas de lenguaje de señas a texto natural en español.

REGLAS:
1. Recibirás palabras en MAYÚSCULAS separadas por espacios (ej: "YO EDWIN HOLA")
2. Debes convertirlas a una oración natural y fluida en español
3. Usa un tono casual y amigable
4. Añade puntuación apropiada (¡! ¿? , .)
5. Puedes añadir conectores naturales (y, pero, que, etc.)
6. NO añadas información que no esté implícita en las palabras
7. Responde SOLO con la oración convertida, sin explicaciones

EJEMPLOS:
- "YO NOMBRE EDWIN" → "Me llamo Edwin"
- "HOLA COMO ESTAR TU" → "¡Hola! ¿Cómo estás?"
- "GRACIAS MUCHO AYUDA" → "¡Muchas gracias por tu ayuda!"
- "YO QUERER AGUA" → "Quiero agua, por favor"
- "DONDE BAÑO" → "¿Dónde está el baño?"
- "NO ENTENDER YO" → "No entiendo"
- "TU BONITO" → "Eres muy bonito/a"
- "YO IR CASA" → "Me voy a casa"
""",
        
        OutputStyle.FORMAL: """Eres un asistente que convierte señas de lenguaje de señas a texto formal en español.

REGLAS:
1. Recibirás palabras en MAYÚSCULAS separadas por espacios
2. Convierte a oraciones formales y respetuosas
3. Usa "usted" en lugar de "tú"
4. Mantén un tono profesional
5. Responde SOLO con la oración convertida

EJEMPLOS:
- "YO NOMBRE EDWIN" → "Mi nombre es Edwin"
- "HOLA COMO ESTAR TU" → "Buenos días, ¿cómo se encuentra usted?"
- "GRACIAS AYUDA" → "Le agradezco mucho su ayuda"
""",
        
        OutputStyle.EXPRESSIVE: """Eres un asistente que convierte señas a texto expresivo y emocional en español.

REGLAS:
1. Recibirás palabras en MAYÚSCULAS
2. Convierte a oraciones con mucha expresividad
3. Usa emojis cuando sea apropiado
4. Añade énfasis y emoción
5. Responde SOLO con la oración convertida

EJEMPLOS:
- "YO NOMBRE EDWIN" → "¡Hola! 👋 Me llamo Edwin, ¡encantado!"
- "HOLA COMO ESTAR TU" → "¡¡Holaaaa!! 😄 ¿Cómo estás?"
- "GRACIAS MUCHO" → "¡Muchísimas gracias! 🙏❤️"
""",

        OutputStyle.MINIMAL: """Convierte señas a texto mínimo en español. Solo la oración, sin extras.

EJEMPLOS:
- "YO NOMBRE EDWIN" → "Soy Edwin"
- "HOLA COMO ESTAR" → "Hola, ¿cómo estás?"
- "GRACIAS" → "Gracias"
"""
    },
    
    "en": {
        OutputStyle.CASUAL: """You are an assistant that converts sign language words to natural English text.

RULES:
1. You'll receive words in UPPERCASE separated by spaces (e.g., "I EDWIN HELLO")
2. Convert them to natural, fluent English sentences
3. Use casual, friendly tone
4. Add appropriate punctuation
5. Respond ONLY with the converted sentence

EXAMPLES:
- "I NAME EDWIN" → "My name is Edwin"
- "HELLO HOW YOU" → "Hey! How are you?"
- "THANK YOU HELP" → "Thanks for your help!"
- "I WANT WATER" → "I'd like some water, please"
""",
        
        OutputStyle.FORMAL: """Convert sign language words to formal English text.
Respond only with the converted sentence.

EXAMPLES:
- "I NAME EDWIN" → "My name is Edwin"
- "HELLO HOW YOU" → "Good day. How are you?"
""",

        OutputStyle.EXPRESSIVE: """Convert signs to expressive English with emojis.
Respond only with the converted sentence.

EXAMPLES:
- "I NAME EDWIN" → "Hey there! 👋 I'm Edwin, nice to meet you!"
- "THANK YOU" → "Thank you so much! 🙏"
""",

        OutputStyle.MINIMAL: """Convert signs to minimal English. Just the sentence.
- "I NAME EDWIN" → "I'm Edwin"
- "HELLO" → "Hello"
"""
    }
}


class SignToTextParser:
    """
    Converts raw sign language predictions to natural text using LLM.
    
    Usage:
        parser = SignToTextParser(api_key="your-deepseek-key")
        
        # Single prediction
        result = parser.parse("YO NOMBRE EDWIN")
        print(result)  # "Me llamo Edwin"
        
        # With context
        result = parser.parse("DONDE BAÑO", context="en un restaurante")
        print(result)  # "¿Dónde está el baño, por favor?"
        
        # Streaming
        for chunk in parser.parse_stream("HOLA COMO ESTAR"):
            print(chunk, end="", flush=True)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "es",
        style: OutputStyle = OutputStyle.CASUAL,
        model: str = "deepseek-chat",
        temperature: float = 0.7
    ):
        """
        Initialize the parser.
        
        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            language: Target language ('es' or 'en')
            style: Output style (casual, formal, expressive, minimal)
            model: DeepSeek model to use
            temperature: LLM temperature for creativity
        """
        self.client = DeepSeekClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=256
        )
        
        self.language = language
        self.style = style if isinstance(style, OutputStyle) else OutputStyle(style)
        
        # Get system prompt
        lang_prompts = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["es"])
        self.system_prompt = lang_prompts.get(self.style, lang_prompts[OutputStyle.CASUAL])
    
    def preprocess(self, raw_signs: str) -> str:
        """
        Preprocess raw sign predictions.
        
        Args:
            raw_signs: Raw prediction string (e.g., "YO_EDWIN_HOLA")
            
        Returns:
            Cleaned string with space-separated words
        """
        # Convert various separators to spaces
        text = raw_signs.replace("_", " ").replace("-", " ")
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Convert to uppercase for consistency
        text = text.upper()
        
        # Remove any special characters except letters and spaces
        text = re.sub(r'[^A-ZÁÉÍÓÚÑÜ\s]', '', text)
        
        return text.strip()
    
    def parse(
        self,
        raw_signs: str,
        context: Optional[str] = None,
        style: Optional[OutputStyle] = None
    ) -> str:
        """
        Convert raw sign predictions to natural text.
        
        Args:
            raw_signs: Raw prediction (e.g., "YO NOMBRE EDWIN")
            context: Optional context for better parsing
            style: Override default style for this call
            
        Returns:
            Natural text string
        """
        # Preprocess
        cleaned = self.preprocess(raw_signs)
        
        if not cleaned:
            return ""
        
        # Build prompt
        prompt = cleaned
        if context:
            prompt = f"[Contexto: {context}] {cleaned}"
        
        # Use custom system prompt if style override
        system = self.system_prompt
        if style and style != self.style:
            lang_prompts = SYSTEM_PROMPTS.get(self.language, SYSTEM_PROMPTS["es"])
            system = lang_prompts.get(style, system)
        
        # Get LLM response
        response = self.client.chat(prompt, system_prompt=system)
        
        # Clean response (remove quotes, extra whitespace)
        result = response.strip().strip('"\'')
        
        return result
    
    def parse_stream(
        self,
        raw_signs: str,
        context: Optional[str] = None
    ):
        """
        Stream the parsing result (generator).
        
        Args:
            raw_signs: Raw prediction string
            context: Optional context
            
        Yields:
            Text chunks as they arrive
        """
        cleaned = self.preprocess(raw_signs)
        
        if not cleaned:
            return
        
        prompt = cleaned
        if context:
            prompt = f"[Contexto: {context}] {cleaned}"
        
        for chunk in self.client.chat_stream(prompt, system_prompt=self.system_prompt):
            yield chunk
    
    def parse_batch(
        self,
        predictions: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Parse multiple predictions.
        
        Args:
            predictions: List of raw prediction strings
            contexts: Optional list of contexts (same length as predictions)
            
        Returns:
            List of natural text strings
        """
        results = []
        contexts = contexts or [None] * len(predictions)
        
        for pred, ctx in zip(predictions, contexts):
            try:
                result = self.parse(pred, context=ctx)
                results.append(result)
            except Exception as e:
                # On error, return cleaned version
                results.append(self.preprocess(pred).title())
        
        return results
    
    def set_style(self, style: OutputStyle):
        """Change output style."""
        self.style = style
        lang_prompts = SYSTEM_PROMPTS.get(self.language, SYSTEM_PROMPTS["es"])
        self.system_prompt = lang_prompts.get(style, lang_prompts[OutputStyle.CASUAL])
    
    def set_language(self, language: str):
        """Change target language."""
        self.language = language
        lang_prompts = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["es"])
        self.system_prompt = lang_prompts.get(self.style, lang_prompts[OutputStyle.CASUAL])


# Convenience function
def parse_signs(
    raw_signs: str,
    api_key: Optional[str] = None,
    language: str = "es",
    style: str = "casual",
    context: Optional[str] = None
) -> str:
    """
    Quick function to parse signs to text.
    
    Args:
        raw_signs: Raw prediction string
        api_key: DeepSeek API key (optional if env var set)
        language: Target language ('es' or 'en')
        style: Output style ('casual', 'formal', 'expressive', 'minimal')
        context: Optional context
        
    Returns:
        Natural text string
        
    Example:
        >>> result = parse_signs("YO NOMBRE EDWIN", language="es")
        >>> print(result)
        "Me llamo Edwin"
    """
    parser = SignToTextParser(
        api_key=api_key,
        language=language,
        style=OutputStyle(style)
    )
    return parser.parse(raw_signs, context=context)


if __name__ == "__main__":
    print("=" * 60)
    print("Sign to Text Parser - Test")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("\n⚠️  DEEPSEEK_API_KEY not set. Running in demo mode.\n")
        
        # Demo without API
        parser_demo = SignToTextParser.__new__(SignToTextParser)
        
        test_cases = [
            "YO NOMBRE EDWIN",
            "HOLA COMO ESTAR TU",
            "GRACIAS MUCHO AYUDA",
            "DONDE BAÑO",
            "YO QUERER AGUA",
            "NO ENTENDER YO",
        ]
        
        print("Sample inputs (would be converted by LLM):")
        for case in test_cases:
            print(f"  • {case}")
        
        print("\n💡 Set DEEPSEEK_API_KEY to test actual conversion.")
        
    else:
        print("\n✅ API key found. Testing parser...\n")
        
        parser = SignToTextParser(api_key=api_key, language="es", style=OutputStyle.CASUAL)
        
        test_cases = [
            ("YO NOMBRE EDWIN", None),
            ("HOLA COMO ESTAR TU", None),
            ("GRACIAS MUCHO AYUDA", None),
            ("DONDE BAÑO", "en un restaurante"),
            ("YO QUERER AGUA", "tengo sed"),
        ]
        
        print("Converting signs to natural text:\n")
        for signs, context in test_cases:
            try:
                result = parser.parse(signs, context=context)
                ctx_str = f" [ctx: {context}]" if context else ""
                print(f"  📝 {signs}{ctx_str}")
                print(f"  ✨ {result}\n")
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
        
        # Test different styles
        print("\n" + "-" * 40)
        print("Testing different styles with 'HOLA COMO ESTAR':\n")
        
        for style in OutputStyle:
            parser.set_style(style)
            try:
                result = parser.parse("HOLA COMO ESTAR")
                print(f"  {style.value.upper():12} → {result}")
            except Exception as e:
                print(f"  {style.value.upper():12} → Error: {e}")
    
    print("\n" + "=" * 60)
