"""
Google Gemini integration for code generation.
This module provides a wrapper around the Google Gemini API for generating cellular automaton rules.
"""

import logging
import os
import google.generativeai as genai
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GeminiCodeGenerator:
    """
    A class to handle code generation using Google's Gemini API.
    """
    
    def __init__(self, model_name="gemini-1.5-pro"):
        """
        Initialize the Gemini code generator with a specified model.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        self.api_key = None
        self.is_configured = False
        self._configure()
        
    def _configure(self):
        """Configure the Gemini API with the API key from environment or secrets."""
        try:
            # Try to get API key from environment
            self.api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
            
            # If not in environment, try to get from Streamlit secrets
            if not self.api_key and hasattr(st, "secrets"):
                # Try to get from openai section in secrets.toml
                if "openai" in st.secrets and "GOOGLE_GEMINI_API_KEY" in st.secrets["openai"]:
                    self.api_key = st.secrets["openai"]["GOOGLE_GEMINI_API_KEY"]
                # Try to get from root level
                elif "GOOGLE_GEMINI_API_KEY" in st.secrets:
                    self.api_key = st.secrets["GOOGLE_GEMINI_API_KEY"]
                
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.is_configured = True
                logger.info(f"Gemini API configured successfully with model {self.model_name}")
            else:
                logger.warning("No Gemini API key found in environment or secrets")
                self.is_configured = False
                
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {str(e)}")
            self.is_configured = False
    
    def generate_code(self, prompt, temperature=0.7, max_tokens=1024):
        """
        Generate code based on the given prompt using Gemini API.
        
        Args:
            prompt: The text prompt describing the code to generate
            temperature: Controls randomness in generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated code as a string or None if API is not configured
        """
        if not self.is_configured:
            logger.warning("Gemini API not configured, cannot generate code")
            return None
            
        try:
            # Format the prompt for code generation
            formatted_prompt = f"""
Generate Python code for a cellular automaton rule based on this description:
"{prompt}"

The code should implement a rule for a cellular automaton with the following requirements:
1. It must create a variable called 'new_grid' initialized as np.zeros_like(grid)
2. It should process each cell in the grid based on its neighbors
3. It can use these constants: DEAD (0), ALIVE (1), DYING (2)
4. It can use the count_neighbors(grid, x, y) function to count a cell's neighbors
5. Return only the Python code, no explanations

Here's the implementation:
```python
# Initialize the new grid
new_grid = np.zeros_like(grid)

"""
            
            # Configure generation parameters
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40
            }
            
            # Create the model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            
            # Generate code
            response = model.generate_content(formatted_prompt)
            
            if response.text:
                # Extract code from the response
                code = response.text
                
                # Clean up the code if it contains markdown code blocks
                if "```python" in code:
                    code = code.split("```python")[1]
                    
                if "```" in code:
                    code = code.split("```")[0]
                
                # Format the code to match the expected structure
                final_code = self._format_code_for_cellular_automaton(code)
                return final_code
            else:
                logger.warning("Empty response from Gemini API")
                return None
                
        except Exception as e:
            logger.error(f"Error generating code with Gemini: {str(e)}")
            return None
    
    def _format_code_for_cellular_automaton(self, code):
        """Format the generated code to work with the cellular automaton framework"""
        # Ensure the code has the necessary structure
        if "new_grid = np.zeros_like(grid)" not in code:
            code = "# Initialize the new grid\nnew_grid = np.zeros_like(grid)\n\n" + code
            
        # Add imports if needed
        if "import numpy" not in code and "import np" not in code:
            code = "import numpy as np\n\n" + code
            
        return code.strip()

# Global instance for reuse
_gemini_generator = None

def get_gemini_generator():
    """Get or create a global GeminiCodeGenerator instance"""
    global _gemini_generator
    if _gemini_generator is None:
        _gemini_generator = GeminiCodeGenerator()
    return _gemini_generator

def generate_rule_code_with_gemini(natural_language_rule):
    """
    Generate code for a cellular automaton rule using Gemini API.
    
    Args:
        natural_language_rule: Natural language description of the rule
        
    Returns:
        Generated code as a string or None if generation failed
    """
    generator = get_gemini_generator()
    if generator.is_configured:
        return generator.generate_code(natural_language_rule)
    return None
