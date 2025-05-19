"""
Local model implementation for code generation using Hugging Face Transformers.
This replaces the OpenAI API calls with a local model to avoid rate limits and costs.
"""

import torch
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model loading status
model_loading_status = {
    "is_loading": False,
    "is_loaded": False,
    "progress": 0,
    "error": None
}

# Lock for thread safety
load_lock = threading.Lock()

class LocalCodeGenerator:
    """
    A class to handle code generation using a local Hugging Face model.
    Uses a singleton pattern to avoid reloading the model multiple times.
    """
    
    # Default model to use for code generation
    DEFAULT_MODEL = "Salesforce/codegen-350M-mono"
    
    # Class variables for singleton pattern
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, model_name="Salesforce/codegen-350M-mono", max_length=1024):
        """Singleton pattern implementation"""
        with load_lock:
            if cls._instance is None:
                cls._instance = super(LocalCodeGenerator, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, model_name="Salesforce/codegen-350M-mono", max_length=1024):
        """
        Initialize the code generator with a specified model.
        Uses a singleton pattern to avoid reloading the model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            max_length: Maximum length of generated text
        """
        # Only initialize once
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
        
        # Don't load the model in __init__, it will be loaded on first use
        # or by the preload_model method
    
    @classmethod
    def preload_model(cls, model_name="Salesforce/codegen-350M-mono"):
        """
        Preload the model in a background thread.
        This can be called early in the application startup.
        """
        def _load_model_thread():
            try:
                # Mark as loading
                with load_lock:
                    model_loading_status["is_loading"] = True
                    model_loading_status["progress"] = 10
                    model_loading_status["error"] = None
                
                logger.info(f"Preloading model {model_name} in background thread")
                
                # Load tokenizer
                with load_lock:
                    model_loading_status["progress"] = 30
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Update progress
                with load_lock:
                    model_loading_status["progress"] = 60
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                ).to(device)
                
                # Store in class variables
                with load_lock:
                    cls._model = model
                    cls._tokenizer = tokenizer
                    model_loading_status["is_loading"] = False
                    model_loading_status["is_loaded"] = True
                    model_loading_status["progress"] = 100
                
                logger.info(f"Model preloaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to preload model: {str(e)}")
                with load_lock:
                    model_loading_status["is_loading"] = False
                    model_loading_status["error"] = str(e)
        
        # Start loading in background thread if not already loaded/loading
        with load_lock:
            if not model_loading_status["is_loading"] and not model_loading_status["is_loaded"]:
                thread = threading.Thread(target=_load_model_thread)
                thread.daemon = True  # Thread will exit when main program exits
                thread.start()
                return True
            return False
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded before use"""
        # If model is already loaded in class variables, use it
        if self.__class__._model is not None and self.__class__._tokenizer is not None:
            self.model = self.__class__._model
            self.tokenizer = self.__class__._tokenizer
            return
        
        # Otherwise load the model
        logger.info(f"Loading model {self.model_name} on {self.device} on demand")
        try:
            with load_lock:
                model_loading_status["is_loading"] = True
                model_loading_status["progress"] = 20
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            with load_lock:
                model_loading_status["progress"] = 60
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Store in class variables for future instances
            self.__class__._model = self.model
            self.__class__._tokenizer = self.tokenizer
            
            with load_lock:
                model_loading_status["is_loading"] = False
                model_loading_status["is_loaded"] = True
                model_loading_status["progress"] = 100
            
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            with load_lock:
                model_loading_status["is_loading"] = False
                model_loading_status["error"] = str(e)
            raise
    
    def generate_code(self, prompt, temperature=0.7, top_p=0.95):
        """
        Generate code based on the given prompt.
        
        Args:
            prompt: The text prompt describing the code to generate
            temperature: Controls randomness in generation
            top_p: Controls diversity via nucleus sampling
            
        Returns:
            Generated code as a string
        """
        try:
            # Ensure model is loaded before use
            self._ensure_model_loaded()
            
            # Format the prompt for code generation
            formatted_prompt = f"""
# Generate Python code for a cellular automaton rule based on this description:
# {prompt}

# Here's the implementation:
def apply_rule(grid):
    # Initialize the new grid
    new_grid = np.zeros_like(grid)
    
"""
            
            # Tokenize the prompt
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Generate code
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=self.max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the code part (remove the prompt)
            code_only = generated_code[len(formatted_prompt):].strip()
            
            # Format the code to match the expected structure
            final_code = self._format_code_for_cellular_automaton(code_only)
            
            return final_code
        
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            # Return a simple template as fallback
            return self._get_fallback_code()
    
    def _format_code_for_cellular_automaton(self, code):
        """Format the generated code to work with the cellular automaton framework"""
        # Ensure the code has the necessary structure
        if "new_grid = np.zeros_like(grid)" not in code:
            code = "new_grid = np.zeros_like(grid)\n" + code
            
        # Add imports if needed
        if "import numpy" not in code and "import np" not in code:
            code = "import numpy as np\n" + code
            
        return code
    
    def _get_fallback_code(self):
        """Return a simple fallback code if generation fails"""
        return """
# Initialize the new grid
new_grid = np.zeros_like(grid)

# Simple rule: cells with exactly 3 neighbors come alive, others die
for x in range(grid.shape[0]):
    for y in range(grid.shape[1]):
        neighbors = count_neighbors(grid, x, y)
        if neighbors == 3:
            new_grid[x, y] = ALIVE
"""

# Create a singleton instance
code_generator = None

def get_code_generator():
    """Get or initialize the code generator singleton"""
    global code_generator
    if code_generator is None:
        try:
            code_generator = LocalCodeGenerator()
        except Exception as e:
            logger.error(f"Failed to initialize code generator: {str(e)}")
            return None
    return code_generator

def generate_rule_code(description):
    """
    Generate cellular automaton rule code from a natural language description.
    
    Args:
        description: Natural language description of the rule
        
    Returns:
        Generated Python code implementing the rule
    """
    generator = get_code_generator()
    if generator:
        return generator.generate_code(description)
    else:
        # Return a simple template if the generator isn't available
        return """
# Initialize the new grid
new_grid = np.zeros_like(grid)

# Loop through each cell
for x in range(grid.shape[0]):
    for y in range(grid.shape[1]):
        # Count neighbors
        neighbors = count_neighbors(grid, x, y)
        
        # Simple rule based on your description
        if grid[x, y] == ALIVE:
            if neighbors < 2 or neighbors > 3:
                new_grid[x, y] = DEAD
            else:
                new_grid[x, y] = ALIVE
        else:
            if neighbors == 3:
                new_grid[x, y] = ALIVE
"""
