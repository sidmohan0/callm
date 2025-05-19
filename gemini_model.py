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
        self.is_configured = False # Initialized to False, configuration is now explicit
        
    def _configure(self):
        """
        Private method to configure the Gemini API with an API key from environment or secrets.
        Sets self.api_key and self.is_configured. Returns True on success, False on failure.
        """
        try:
            api_key_from_env = os.environ.get("GOOGLE_GEMINI_API_KEY")
            
            api_key_from_secrets = None
            if not api_key_from_env and hasattr(st, "secrets"):
                if "openai" in st.secrets and "GOOGLE_GEMINI_API_KEY" in st.secrets["openai"]:
                    api_key_from_secrets = st.secrets["openai"]["GOOGLE_GEMINI_API_KEY"]
                elif "GOOGLE_GEMINI_API_KEY" in st.secrets:
                    api_key_from_secrets = st.secrets["GOOGLE_GEMINI_API_KEY"]
            
            key_to_try = api_key_from_env or api_key_from_secrets

            if key_to_try:
                genai.configure(api_key=key_to_try) # Try configuring first
                self.api_key = key_to_try           # If successful, store it
                self.is_configured = True
                logger.info(f"Gemini API configured successfully from env/secrets with model {self.model_name}")
                return True
            else:
                # This case means no key was found in env or secrets. It's not an error, just not configured yet.
                # logger.warning("No Gemini API key found in environment or secrets during initial configuration attempt.") 
                self.is_configured = False
                return False
            
        except Exception as e:
            logger.error(f"Error during initial Gemini API configuration from env/secrets: {str(e)}")
            self.is_configured = False
            return False

    def attempt_initial_configuration(self):
        """Attempts to configure the API key from environment/secrets. Returns True/False."""
        return self._configure()

    def configure_with_user_key(self, user_api_key):
        """
        Attempts to configure the API with a user-provided key.
        Sets self.api_key and self.is_configured. Returns True on success, False on failure.
        """
        if not user_api_key or not isinstance(user_api_key, str) or not user_api_key.strip():
            logger.warning("User-provided API key is empty or invalid.")
            self.is_configured = False # Ensure this is set if key is bad before trying
            return False
        
        try:
            genai.configure(api_key=user_api_key.strip()) # Try configuring first
            self.api_key = user_api_key.strip()           # If successful, store it
            self.is_configured = True
            logger.info(f"Gemini API configured successfully with user-provided key using model {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error configuring Gemini API with user-provided key: {str(e)}")
            self.is_configured = False
            # self.api_key = None # Optionally clear the bad key if it's now known to be invalid
            return False

    def generate_code(self, prompt, temperature=0.6, max_tokens=2048):
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
            formatted_prompt = f"""
You are a Python expert specializing in cellular automata. Write Python code for a cellular automaton rule that implements this description:
"{prompt}"

FOLLOW THESE RULES EXACTLY:
1. Your code MUST create a variable called 'new_grid' initialized as np.zeros_like(grid)
2. Loop through each cell in the grid using nested for loops over x and y coordinates
3. Use count_neighbors(grid, x, y) to count the number of ALIVE neighbors (returns an integer 0-8)
4. Use these constants for cell states: DEAD (0), ALIVE (1), DYING (2)
5. Return ONLY working Python code with no explanations or markdown

AVAILABLE VARIABLES:
- grid: 2D numpy array containing the current state of all cells
- DEAD, ALIVE, DYING: Integer constants representing cell states
- count_neighbors(grid, x, y): Function that returns number of ALIVE neighbors

CODE TEMPLATE:
```python
# Initialize the new grid
new_grid = np.zeros_like(grid)

# Loop through each cell in the grid
for x in range(grid.shape[0]):
    for y in range(grid.shape[1]):
        # Get current cell state
        current_state = grid[x, y]
        
        # Count live neighbors
        neighbors = count_neighbors(grid, x, y)
        
        # YOUR RULE LOGIC HERE
        # Example: if current_state == DEAD and neighbors == 3:
        #             new_grid[x, y] = ALIVE
```

Provide only the complete, working Python code:
"""
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40
            }
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            
            response = model.generate_content(formatted_prompt)
            
            if response.prompt_feedback:
                logger.info(f"Gemini API Prompt Feedback: {response.prompt_feedback}")
            
            if response.text:
                logger.info(f"Raw Gemini API Response Text: {response.text}")
                code = response.text
                
                if "```python" in code:
                    code = code.split("```python")[1]
                    
                if "```" in code:
                    code = code.split("```")[0]
                
                final_code = self._format_code_for_cellular_automaton(code)
                return final_code
            else:
                logger.warning("Empty response from Gemini API")
                return None
                
        except Exception as e:
            if hasattr(e, 'message'):
                logger.error(f"Error generating code with Gemini: {e.message}")
            elif hasattr(e, 'args') and e.args:
                 logger.error(f"Error generating code with Gemini: {e.args[0]}")
            else:
                logger.error(f"Error generating code with Gemini: {str(e)}")
            return None

    def _format_code_for_cellular_automaton(self, code):
        """Format the generated code to work with the cellular automaton framework"""
        if "new_grid = np.zeros_like(grid)" not in code:
            code = "# Initialize the new grid\nnew_grid = np.zeros_like(grid)\n\n" + code
            
        if "import numpy" not in code and "import np" not in code:
            code = "import numpy as np\n\n" + code
            
        return code.strip()
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
