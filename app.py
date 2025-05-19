import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time
import ast
import logging
from matplotlib.colors import ListedColormap
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define rate limit constant for Gemini API calls
RATE_LIMIT_SECONDS = 10

#############################################################
# CONFIGURATION AND CONSTANTS                               #
#############################################################

# Define a configuration class to encapsulate all settings
class Config:
    """
    Configuration class to encapsulate all application settings.
    This makes the code more modular and easier to test by removing
    global variables and Streamlit widget dependencies.
    """
    # Cell state definitions
    DEAD = 0
    ALIVE = 1
    DYING = 2
    
    # Grid size limits
    MIN_GRID_SIZE = 20
    MAX_GRID_SIZE = 200  # Upper limit to prevent performance issues
    DEFAULT_GRID_SIZE = 100
    WARNING_THRESHOLD = 150  # Show warning for grid sizes above this threshold
    
    # Simulation constants
    DEFAULT_STEPS = 100
    MIN_STEPS = 1
    MAX_STEPS = 500
    
    # Delay settings
    MIN_DELAY = 0.0
    MAX_DELAY = 1.0
    DEFAULT_DELAY = 0.1
    DELAY_STEP = 0.05
    
    # Available rule sets
    RULE_SETS = ["Custom LLM Rule", "Brian's Brain", "Seeds", "Noise-Life"]
    DEFAULT_RULE = "Custom LLM Rule"
    
    # Visualization settings
    COLOR_PALETTE = {
        'dead': '#000000',  # Black
        'alive': '#66ff66',  # Soft green
        'dying': '#ff6666',  # Red-orange
        'highlight': '#66ccff'  # Sky blue
    }
    
    def __init__(self):
        """Initialize with default values"""
        # Runtime configuration (can be changed by UI)
        self.grid_size = self.DEFAULT_GRID_SIZE
        self.num_steps = self.DEFAULT_STEPS
        self.delay = self.DEFAULT_DELAY
        self.rule_choice = self.DEFAULT_RULE
        
        # Create colormap from palette
        self.pixel_palette = [self.COLOR_PALETTE['dead'], 
                             self.COLOR_PALETTE['alive'], 
                             self.COLOR_PALETTE['dying'], 
                             self.COLOR_PALETTE['highlight']]
        self.cmap = ListedColormap(self.pixel_palette)
    
    def update_from_ui(self):
        """Update configuration from Streamlit UI widgets"""
        # Grid size with validation
        self.grid_size = st.sidebar.slider(
            "Grid Size", 
            min_value=self.MIN_GRID_SIZE, 
            max_value=self.MAX_GRID_SIZE, 
            value=self.DEFAULT_GRID_SIZE,
            help=f"Size of the cellular automaton grid. Values above {self.WARNING_THRESHOLD} may impact performance."
        )
        
        # Show warning for large grid sizes
        if self.grid_size > self.WARNING_THRESHOLD:
            st.sidebar.warning(f"⚠️ Grid sizes above {self.WARNING_THRESHOLD} may cause performance issues on some devices.")
        
        # Other configuration options
        self.num_steps = st.sidebar.slider("Number of Steps", 
                                        min_value=self.MIN_STEPS, 
                                        max_value=self.MAX_STEPS, 
                                        value=self.DEFAULT_STEPS)
        
        self.delay = st.sidebar.slider("Delay Between Steps (seconds)", 
                                     self.MIN_DELAY, 
                                     self.MAX_DELAY, 
                                     self.DEFAULT_DELAY, 
                                     step=self.DELAY_STEP)
        
        # Rule descriptions for tooltips
        rule_descriptions = {
            "Brian's Brain": "A cellular automaton with three states: alive (blue), dying (red), and dead (black). "
                           "Cells come alive with exactly 2 alive neighbors, alive cells always start dying, "
                           "and dying cells always die.",
            "Seeds": "A cellular automaton where cells are born if they have exactly 2 neighbors, "
                    "otherwise they die. Creates beautiful fractal-like patterns.",
            "Noise-Life": "A variant of Conway's Game of Life with random noise. "
                        "Similar rules to Conway but with a 1% chance of random state changes.",
            "Custom LLM Rule": "Create your own cellular automaton rules using natural language. "
                           "Describe the rules in plain English and let AI generate the code."
        }
        
        # Use radio buttons instead of selectbox for rule selection
        st.sidebar.subheader("Select Rule Set")
        self.rule_choice = st.sidebar.radio(
            "Rule Type",  # Non-empty label for accessibility
            options=self.RULE_SETS,
            index=self.RULE_SETS.index(self.DEFAULT_RULE),
            help="Select a cellular automaton rule set to simulate",
            label_visibility="collapsed"  # Hide the label since we have a subheader
        )
        
        # Show description tooltip for the selected rule
        st.sidebar.info(rule_descriptions[self.rule_choice])

# Create a global configuration instance
config = Config()

# For backward compatibility, define these at module level
# This allows existing code to work without major changes
DEAD = config.DEAD
ALIVE = config.ALIVE
DYING = config.DYING
cmap = config.cmap

#############################################################
# UI COMPONENTS - SIDEBAR CONTROLS                         #
#############################################################

# Add GitHub badge at the top of the sidebar
st.sidebar.markdown(
    "<div style='text-align: center;'>"
    "<a href='https://github.com/sidmohan0/callm' target='_blank'>"
    "<img src='https://img.shields.io/badge/GitHub-Repository-181717?logo=github' alt='GitHub Repo'>"
    "</a>"
    "</div>", 
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Update configuration from UI
config.update_from_ui()

# GitHub badge moved to the bottom of the sidebar

# For convenience, extract commonly used values to local variables
grid_size = config.grid_size
num_steps = config.num_steps
delay = config.delay
rule_choice = config.rule_choice

# Initialize grid
def initialize_grid(size, rule_type=None, cfg=None):
    """
    Initialize a grid with appropriate patterns based on rule type.
    
    Args:
        size: Size of the grid (size x size)
        rule_type: Type of cellular automaton rule, defaults to current rule_choice if None
        cfg: Configuration object, defaults to global config if None
        
    Returns:
        Initialized grid with appropriate patterns for the specified rule
    """
    # Use provided config or global config
    cfg = cfg or config
    
    # If rule_type is not provided, use the config's rule_choice
    if rule_type is None:
        rule_type = cfg.rule_choice
    
    # Create an empty grid (all DEAD)
    grid = np.zeros((size, size), dtype=int)
    
    # Different initialization patterns based on rule type
    if rule_type == "Brian's Brain":
        # Brian's Brain works best with patterns that create cells with exactly 2 neighbors
        return initialize_brians_brain(grid, size)
    elif rule_type == "Seeds":
        # Seeds works best with scattered patterns
        return initialize_seeds(grid, size)
    elif rule_type == "Noise-Life":
        # Noise-Life works well with random patterns
        return initialize_noise_life(grid, size)
    else:
        # For custom rules, use a general-purpose pattern
        return initialize_general(grid, size)

# Specialized initialization for Brian's Brain
def initialize_brians_brain(grid, size):
    # Start point in the top-left corner with some offset
    start_x, start_y = 5, 5
    
    # Create a diagonal line of cells - this acts as a "fuse"
    fuse_length = min(size // 4, 30)  # Limit the length for very large grids
    
    # Place the expanding pattern
    for i in range(fuse_length):
        # Create a diagonal line
        grid[start_x + i, start_y + i] = ALIVE
        
        # Add cells that will create exactly 2 neighbors for adjacent cells
        if i > 0 and i < fuse_length - 1:
            # Add cells above and below the diagonal
            grid[start_x + i, start_y + i - 1] = ALIVE
            grid[start_x + i - 1, start_y + i] = ALIVE
    
    # Add a small cluster at the start to ensure continuous propagation
    cluster_size = 4
    for dx in range(cluster_size):
        for dy in range(cluster_size):
            if dx + dy < cluster_size and dx > 0 and dy > 0:
                grid[start_x + dx, start_y + dy] = ALIVE
    
    # Add some random "sparks" near the fuse
    spark_count = 20
    for _ in range(spark_count):
        # Random position near the fuse
        spark_x = start_x + np.random.randint(0, fuse_length + 10)
        spark_y = start_y + np.random.randint(0, fuse_length + 10)
        
        # Ensure it's within bounds
        if spark_x < size - 2 and spark_y < size - 2:
            # Create a small 2x2 pattern that ensures propagation
            grid[spark_x, spark_y] = ALIVE
            grid[spark_x + 1, spark_y] = ALIVE
            grid[spark_x, spark_y + 1] = ALIVE
    
    return grid

# Specialized initialization for Seeds
def initialize_seeds(grid, size):
    # Seeds needs scattered points with specific spacing
    # Create a grid with ~10% of cells alive, but in a pattern
    # that ensures some cells will have exactly 2 neighbors
    
    # Create a few seed clusters
    num_clusters = 5
    for _ in range(num_clusters):
        # Random cluster center
        center_x = np.random.randint(size // 4, 3 * size // 4)
        center_y = np.random.randint(size // 4, 3 * size // 4)
        
        # Create a cluster of seeds
        radius = np.random.randint(5, 15)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = center_x + dx, center_y + dy
                if 0 <= x < size and 0 <= y < size:
                    # Create a pattern where some cells will have exactly 2 neighbors
                    if (dx**2 + dy**2 <= radius**2) and np.random.random() < 0.3:
                        grid[x, y] = ALIVE
    
    return grid

# Specialized initialization for Noise-Life
def initialize_noise_life(grid, size):
    # Noise-Life works well with random patterns
    # Create a grid with ~20% of cells alive randomly
    for x in range(size):
        for y in range(size):
            if np.random.random() < 0.2:
                grid[x, y] = ALIVE
    
    return grid

# General-purpose initialization for custom rules
def initialize_general(grid, size):
    # Create a diverse set of patterns that should work with most rules
    
    # 1. Add a glider in the top-left
    glider = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ]
    offset = 10
    for i in range(len(glider)):
        for j in range(len(glider[0])):
            if glider[i][j] == 1:
                grid[offset + i, offset + j] = ALIVE
    
    # 2. Add a small oscillator in the bottom-right
    blinker = [[ALIVE, ALIVE, ALIVE]]
    br_offset = size - 15
    for i in range(len(blinker)):
        for j in range(len(blinker[0])):
            grid[br_offset + i, br_offset + j] = blinker[i][j]
    
    # 3. Add some random cells throughout
    for _ in range(size * size // 20):  # Fill about 5% randomly
        x, y = np.random.randint(0, size), np.random.randint(0, size)
        grid[x, y] = ALIVE
    
    return grid

def count_neighbors(grid, x, y):
    # Get grid dimensions
    height, width = grid.shape
    
    # Define the bounds for the neighborhood, ensuring we don't go out of bounds
    x_min = max(x-1, 0)
    x_max = min(x+2, height)
    y_min = max(y-1, 0)
    y_max = min(y+2, width)
    
    # Get the neighborhood
    neighbors = grid[x_min:x_max, y_min:y_max]
    
    # Count ALIVE cells and subtract the center cell if it's ALIVE
    return np.count_nonzero(neighbors == ALIVE) - (grid[x, y] == ALIVE)

#############################################################
# CELLULAR AUTOMATA RULE IMPLEMENTATIONS                    #
#############################################################

def brians_brain_step(grid):
    """
    Implements Brian's Brain cellular automaton rules:
    1. DEAD cells with exactly 2 ALIVE neighbors become ALIVE
    2. ALIVE cells always become DYING
    3. DYING cells always become DEAD
    
    Args:
        grid: 2D numpy array representing the current state
        
    Returns:
        Updated grid after applying the rules
    """
    new_grid = np.zeros_like(grid)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            state = grid[x, y]
            neighbors = count_neighbors(grid, x, y)
            if state == DEAD and neighbors == 2:
                new_grid[x, y] = ALIVE
            elif state == ALIVE:
                new_grid[x, y] = DYING
            elif state == DYING:
                new_grid[x, y] = DEAD
    return new_grid

def seeds_step(grid):
    """
    Implements Seeds cellular automaton rules:
    1. DEAD cells with exactly 2 ALIVE neighbors become ALIVE
    2. All other cells become/stay DEAD
    
    Args:
        grid: 2D numpy array representing the current state
        
    Returns:
        Updated grid after applying the rules
    """
    new_grid = np.zeros_like(grid)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == DEAD:
                if count_neighbors(grid, x, y) == 2:
                    new_grid[x, y] = ALIVE
    return new_grid

def noise_life_step(grid):
    """
    Implements Noise-Life cellular automaton rules:
    1. Similar to Conway's Game of Life but with random noise
    2. ALIVE cells with fewer than 2 or more than 3 ALIVE neighbors die
    3. DEAD cells with exactly 3 ALIVE neighbors become ALIVE
    4. 1% chance of random state change for any cell
    
    Args:
        grid: 2D numpy array representing the current state
        
    Returns:
        Updated grid after applying the rules
    """
    new_grid = np.copy(grid)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            neighbors = count_neighbors(grid, x, y)
            if grid[x, y] == ALIVE:
                if neighbors < 2 or neighbors > 3:
                    new_grid[x, y] = DEAD
            else:
                if neighbors == 3:
                    new_grid[x, y] = ALIVE
            if np.random.rand() < 0.01:
                new_grid[x, y] = np.random.choice([DEAD, ALIVE])
    return new_grid

#############################################################
# CUSTOM RULE EXECUTION                                     #
#############################################################

def run_custom_rule(grid, custom_code):
    """
    Safely executes user-provided custom rule code.
    
    Args:
        grid: The current grid state
        custom_code: Python code string defining cellular automaton rules
        
    Returns:
        Updated grid after applying the custom rules, or original grid if execution failed
    """
    try:
        # Strip any leading/trailing whitespace
        custom_code = custom_code.strip()
        
        # Check if the code is empty
        if not custom_code:
            st.error("Custom rule code cannot be empty. Using default rule.")
            return grid
        
        # Ensure new_grid is initialized
        if "new_grid" not in custom_code:
            st.error("Custom rule must define 'new_grid'. Using default rule.")
            return grid
            
        # Set up the execution environment with safe variables only
        local_vars = {
            "grid": grid, 
            "np": np, 
            "count_neighbors": count_neighbors, 
            "DEAD": DEAD, 
            "ALIVE": ALIVE, 
            "DYING": DYING,
            "new_grid": np.zeros_like(grid)  # Pre-initialize new_grid as a safety measure
        }
        
        # Execute the custom code in a restricted environment (no globals)
        exec(custom_code, {}, local_vars)
        
        # Verify new_grid exists and has the right shape
        if 'new_grid' not in local_vars:
            st.error("Custom rule did not define 'new_grid'. Using default rule.")
            return grid
            
        if local_vars['new_grid'].shape != grid.shape:
            st.error(f"Custom rule produced grid with wrong shape: {local_vars['new_grid'].shape} vs {grid.shape}. Using default rule.")
            return grid
            
        return local_vars['new_grid']
    except SyntaxError as e:
        line_num = e.lineno if hasattr(e, 'lineno') else '?'
        st.error(f"Syntax error in custom rule at line {line_num}: {e}")
        st.info("Check for missing colons, parentheses, or indentation issues.")
        return grid
    except Exception as e:
        st.error(f"Error in custom rule: {str(e)}")
        st.info("Make sure your code correctly defines 'new_grid' and uses the provided variables.")
        return grid

#############################################################
# AI MODEL INTEGRATION                                      #
#############################################################

# Initialize session state for message management
if 'show_model_messages' not in st.session_state:
    st.session_state.show_model_messages = True

# Initialize model availability flag
gemini_available = False

# Initialize session state for API key configuration if not already set
if 'user_api_key_configured' not in st.session_state:
    st.session_state.user_api_key_configured = False

# Import the Gemini model for code generation
if st.session_state.show_model_messages:
    try:
        from gemini_model import generate_rule_code_with_gemini, get_gemini_generator
        gemini_generator = get_gemini_generator()
        
        # Attempt initial configuration (from env/secrets)
        gemini_generator.attempt_initial_configuration()
        gemini_available = gemini_generator.is_configured
        
        # If the user previously configured an API key in this session, update gemini_available
        if st.session_state.user_api_key_configured and 'user_gemini_api_key' in st.session_state:
            if not gemini_available: # Only try if not already configured
                gemini_available = gemini_generator.configure_with_user_key(st.session_state.user_gemini_api_key)
        
        # Add UI for user to input API key if not configured
        if not gemini_available:
            st.sidebar.warning("⚠️ Google Gemini API not configured")
            st.sidebar.info("You can enter your API key below or add GOOGLE_GEMINI_API_KEY to .streamlit/secrets.toml")
            
            # Create a form for API key input
            with st.sidebar.form("gemini_api_key_form"):
                user_api_key = st.text_input("Google Gemini API Key", type="password", help="Get a key at https://ai.google.dev/")
                submit_key = st.form_submit_button("Connect")
                
                if submit_key and user_api_key:
                    # Try to configure with user-provided key
                    if gemini_generator.configure_with_user_key(user_api_key):
                        # Set the flag that user has configured an API key successfully
                        st.session_state.user_api_key_configured = True
                        # Store the API key in session state for reuse
                        st.session_state.user_gemini_api_key = user_api_key
                        # Update availability flags
                        gemini_available = True
                        any_model_available = True
                        # Show success message
                        st.success("✅ API key configured successfully!")
                        # Store in session state that we're using a user-provided key
                        st.session_state.user_provided_gemini_key = True
                        # Force a complete rerun of the app
                        st.experimental_rerun()
                    else:
                        st.error("❌ Invalid API key or connection error. Please try again.")
        else:
            # API is configured, show success message
            st.sidebar.success(f"✅ Google Gemini API connected and ready")
            st.sidebar.markdown(f"Model: [{gemini_generator.model_name}](https://ai.google.dev/models/{gemini_generator.model_name})")
            
            # If using a user-provided key, show a note and option to reset
            if st.session_state.get('user_provided_gemini_key', False):
                st.sidebar.info("Using your provided API key for this session")
                if st.sidebar.button("Reset API Key"):
                    # Clear the user-provided key flag
                    if 'user_provided_gemini_key' in st.session_state:
                        del st.session_state['user_provided_gemini_key']
                    # Reset all configuration flags
                    gemini_generator.is_configured = False
                    gemini_available = False
                    any_model_available = False
                    # Clear session state variables
                    st.session_state.user_api_key_configured = False
                    if 'user_gemini_api_key' in st.session_state:
                        del st.session_state.user_gemini_api_key
                    if 'user_provided_gemini_key' in st.session_state:
                        del st.session_state.user_provided_gemini_key
                    # Force a complete rerun of the app
                    st.experimental_rerun()
    except ImportError as e:
        st.sidebar.error(f"⚠️ Error importing Gemini model: {str(e)}")
        st.sidebar.info("To enable rule generation, install the required dependency: pip install google-generativeai")
        gemini_available = False
    except Exception as e:
        st.sidebar.error(f"⚠️ Error setting up Gemini: {str(e)}")
        gemini_available = False
else:
    # Don't show messages but still try to import and configure
    try:
        from gemini_model import generate_rule_code_with_gemini, get_gemini_generator
        gemini_generator = get_gemini_generator()
        
        # Try to configure from environment/secrets first
        gemini_generator.attempt_initial_configuration()
        
        # If the user previously configured an API key in this session, use it
        if not gemini_generator.is_configured and st.session_state.user_api_key_configured and 'user_gemini_api_key' in st.session_state:
            gemini_generator.configure_with_user_key(st.session_state.user_gemini_api_key)
            
        gemini_available = gemini_generator.is_configured
    except:
        gemini_available = False

# Set any_model_available based on gemini_available and user_api_key_configured
any_model_available = gemini_available or st.session_state.user_api_key_configured

# Add a button to reset loading messages
if st.sidebar.button("Reset Loading Messages", help="Click to clear and reset the model loading messages"):
    st.session_state.show_model_messages = False
    st.experimental_rerun()

# GitHub badge will be added at the top of the sidebar

# Rename the imported function to avoid namespace conflicts
if 'generate_rule_code' in locals():
    local_generate_rule_code = generate_rule_code

def generate_rule_code(natural_language_rule):
    """
    Generates Python code for a cellular automaton rule based on a natural language description.
    
    Uses Google Gemini API to translate the natural language description into executable Python code.
    
    Args:
        natural_language_rule: A string describing the desired cellular automaton rule
        
    Returns:
        A string containing Python code implementing the rule, or None if generation failed
    """
    if gemini_available:
        try:
            gemini_code = generate_rule_code_with_gemini(natural_language_rule)
            if gemini_code:
                logger.info("Successfully generated code with Gemini")
                return gemini_code
            else:
                logger.warning("Gemini code generation failed")
        except Exception as e:
            logger.error(f"Error with Gemini code generation: {str(e)}")
    
    # If we get here, Gemini failed or is not available
    st.error("⚠️ Unable to generate code from description.")
    st.info("Please check your Gemini API configuration or try again with a different description.")
    return None

def _generate_fallback_code(description):
    """Generate a simple fallback implementation when the model isn't available"""
    # Create a simplified rule based on Conway's Game of Life
    return f"""
# Initialize the new grid - always start with this line
new_grid = np.zeros_like(grid)  # Creates a grid of zeros with the same shape as the input grid

# Rule based on description: {description}
# This is a simplified implementation of Conway's Game of Life

# Loop through each cell in the grid
for x in range(grid.shape[0]):
    for y in range(grid.shape[1]):
        # Count how many ALIVE neighbors this cell has
        neighbors = count_neighbors(grid, x, y)
        
        # Conway's Game of Life rules:
        # 1. Any live cell with 2 or 3 live neighbors survives
        # 2. Any dead cell with exactly 3 live neighbors becomes alive
        # 3. All other cells die or stay dead
        
        if grid[x, y] == ALIVE:  # If the cell is currently alive
            if neighbors in [2, 3]:
                new_grid[x, y] = ALIVE  # Cell survives
            else:
                new_grid[x, y] = DEAD   # Cell dies
        else:  # If the cell is currently dead
            if neighbors == 3:
                new_grid[x, y] = ALIVE  # Cell becomes alive
            # Otherwise it stays dead (new_grid already initialized to DEAD)
"""

#############################################################
# MAIN APPLICATION LOGIC                                    #
#############################################################

# Application class to encapsulate the main logic
class CellularAutomataApp:
    """
    Main application class for the Cellular Automata Viewer.
    Encapsulates the application logic and state management.
    """
    def __init__(self, cfg):
        """
        Initialize the application with the given configuration.
        
        Args:
            cfg: Configuration object with application settings
        """
        self.config = cfg
        
        # Rule map - maps rule names to their implementation functions
        self.rule_map = {
            "Brian's Brain": brians_brain_step,
            "Seeds": seeds_step,
            "Noise-Life": noise_life_step
        }
        
        # Initialize the grid based on the selected rule
        self.grid = initialize_grid(self.config.grid_size, cfg=self.config)
        
        # Set the rule function based on the selected rule
        self.custom_code = ""
        
    def get_rule_function(self):
        """
        Get the appropriate rule function based on the current configuration.
        
        Returns:
            A function that takes a grid and returns the next state
        """
        if self.config.rule_choice == "Custom LLM Rule":
            return lambda g: run_custom_rule(g, self.custom_code)
        else:
            return self.rule_map[self.config.rule_choice]
            
    def set_custom_code(self, code):
        """Set the custom rule code"""
        self.custom_code = code

# Create the application instance
app = CellularAutomataApp(config)

# Application title
st.title("CALLM: AI-Powered Cellular Automata Explorer")

# GitHub badge below the title
st.markdown(
    "<div>"
    "<a href='https://github.com/sidmohan0/callm' target='_blank'>"
    "<img src='https://img.shields.io/badge/GitHub-Repository-181717?logo=github' alt='GitHub Repo'>"
    "</a>"
    "</div>", 
    unsafe_allow_html=True
)

# Informative subheading with context
st.markdown("""
    Cellular automata are mathematical models where cells on a grid evolve through time based on simple rules.
    Each cell's new state depends on its current state and the states of neighboring cells, creating complex patterns from simple rules.
    
    **CALLM** lets you explore classic cellular automata like Brian's Brain and Seeds, or create your own rules using natural language.
    Watch as complex patterns emerge, stabilize, or repeat in real-time!
""")

# For backward compatibility
grid = app.grid

# Handle custom rule input if selected
if config.rule_choice == "Custom LLM Rule":
    st.markdown("### Create Your Own Cellular Automaton Rule")
    
    # Step-by-step instructions
    with st.expander("📋 How to create custom rules", expanded=True):
        st.markdown("""
        **Creating your own cellular automaton is a 3-step process:**
        
        1. **Describe your rule** in plain English in the text box below
           - Example: "Cells with exactly 2 neighbors come alive, all others die"
           - Be specific about conditions for cells becoming alive, dying, or staying the same
        
        2. **Generate code** by clicking the button (requires local AI model)
           - The AI will translate your description into Python code
           - **The generated code is automatically applied to your simulation**
           - No need to copy/paste - it's ready to use immediately!
        
        3. **Edit the code** if needed to fine-tune your rule
           - The code must create a variable called `new_grid`
           - You can use these constants: `DEAD (0)`, `ALIVE (1)`, `DYING (2)`
           - Use `count_neighbors(grid, x, y)` to count a cell's neighbors
        """)
    
    # Default custom code template with better comments
    default_custom_code = """
# Initialize the new grid - this line is required
new_grid = np.zeros_like(grid)  # Creates a grid of zeros with the same shape as the input grid

# Loop through each cell in the grid
for x in range(grid.shape[0]):
    for y in range(grid.shape[1]):
        # Count how many ALIVE neighbors this cell has (returns a number from 0-8)
        neighbors = count_neighbors(grid, x, y)
        
        # Conway's Game of Life rules (modify these to create your own rule):
        if grid[x, y] == ALIVE:  # If the cell is currently alive
            if neighbors in [2, 3]:
                new_grid[x, y] = ALIVE  # Cell survives
            else:
                new_grid[x, y] = DEAD   # Cell dies
        else:  # If the cell is currently dead
            if neighbors == 3:
                new_grid[x, y] = ALIVE  # Cell becomes alive
            # Otherwise it stays dead

# Available states: 
# - DEAD (0): Cell is inactive
# - ALIVE (1): Cell is active
# - DYING (2): Cell is in transition (used in Brian's Brain)
"""
    
    # Step 1: Describe the rule
    st.markdown("#### Step 1: Describe your rule")
    natural_desc = st.text_input(
        "Describe your rule in plain English:", 
        "Cells surrounded by 3 neighbors come alive, others die.",
        help="Be specific about when cells should become alive, die, or enter other states"
    )
    
    # Initialize with default if no custom code has been set yet
    if not hasattr(app, 'custom_code') or not app.custom_code:
        app.set_custom_code(default_custom_code)
    
    # Step 2: Generate code
    st.markdown("#### Step 2: Generate code from description")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Generate button with conditional display based on model availability
        generate_button = st.button(
            "🧠 Generate Code", 
            disabled=not any_model_available,
            help="Google Gemini API not configured" if not any_model_available else 
                 "Click to generate code from your description"
        )
    
    with col2:
        if not gemini_available:
            st.warning("⚠️ Google Gemini API not available. Please check your API key configuration.")
            st.info("Add GOOGLE_GEMINI_API_KEY to .streamlit/secrets.toml or as an environment variable")


    # Try to generate code from description if button is clicked and any model is available
    if generate_button and natural_desc.strip():
        if any_model_available:
            current_time = time.time()
            last_call_time = st.session_state.get('last_gemini_call_time', 0)

            if current_time - last_call_time < RATE_LIMIT_SECONDS:
                remaining_time = RATE_LIMIT_SECONDS - (current_time - last_call_time)
                st.warning(f"⏳ Please wait {remaining_time:.1f} seconds before trying again.")
            else:
                try:
                    # Using Gemini API for code generation
                    with st.spinner("🔄 Generating rule code using Google Gemini API... This may take a moment."):
                        # Generate code
                        generated_code = generate_rule_code(natural_desc)
                        st.session_state['last_gemini_call_time'] = time.time() # Update timestamp
                        
                        if generated_code is not None:
                            # Show success message
                            st.success(f"✅ Rule generated by Google Gemini and applied automatically!")
                            
                            # Show the generated code in a collapsible section
                            with st.expander("View generated code", expanded=False):
                                st.code(generated_code, language='python')
                                st.caption("This code has already been applied to the rule editor below.")
                            
                            # Apply the generated code to the rule editor
                            app.set_custom_code(generated_code)
                            
                        else:
                            st.error("⚠️ Could not generate code. Try a more specific description or edit the code manually.")

                except Exception as e:
                    st.error(f"❌ Error generating code: {str(e)}")
                    st.info("You can still create a rule by editing the code manually below.")
        else:
            st.error("❌ Google Gemini API is not available.")
            st.info("To enable rule generation, configure the Google Gemini API key.")
            st.info("Add GOOGLE_GEMINI_API_KEY to .streamlit/secrets.toml or as an environment variable")
    
    # Step 3: Edit code
    st.markdown("#### Step 3: Edit your rule code")
    st.info("💡 This is your active rule code - changes here are applied immediately to the simulation")
    
    # Store the previous code to detect changes
    if 'previous_code' not in st.session_state:
        st.session_state.previous_code = app.custom_code or default_custom_code
    
    # Text area for code editing
    new_code = st.text_area(
        "Active Rule Code:", 
        height=250, 
        value=app.custom_code or default_custom_code,
        help="Your code must create a variable called 'new_grid' that will be the next state of the grid"
    )
    
    # Apply the code and show confirmation if changed
    app.set_custom_code(new_code)
    
    # Check if code has changed
    if new_code != st.session_state.previous_code:
        st.success("✅ Rule code updated and applied to simulation")
        st.session_state.previous_code = new_code

# Get the appropriate rule function based on the current configuration
rule_fn = app.get_rule_function()

#############################################################
# ANIMATION AND UI CONTROL LOGIC                            #
#############################################################

# Animation state management class
class AnimationState:
    """
    Class to manage the animation state and pattern detection.
    This encapsulates all the animation logic in one place.
    """
    def __init__(self, app_instance, config_instance):
        """Initialize animation state with application and configuration"""
        self.app = app_instance
        self.config = config_instance
        
        # Initialize state if not already done
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.step = 0
            st.session_state.grid = self.app.grid.copy()
            st.session_state.paused = False
            # Auto-start the simulation by default for better UX
            st.session_state.auto_advance = True
            
            # Pattern detection to prevent infinite loops
            st.session_state.pattern_history = []  # Store recent grid states
            st.session_state.stable_detected = False
            st.session_state.repeating_detected = False
            st.session_state.history_length = 10  # Number of previous states to keep
            
            # Initialize speed multiplier
            st.session_state.speed_multiplier = 1.0
    
    def reset(self):
        """Reset the simulation to initial state"""
        # Store current auto-advance state
        was_auto_advancing = st.session_state.auto_advance
        
        # Reset simulation state
        st.session_state.step = 0
        st.session_state.grid = initialize_grid(self.config.grid_size, cfg=self.config)
        st.session_state.paused = False
        st.session_state.pattern_history = []  # Reset pattern history
        st.session_state.stable_detected = False
        st.session_state.repeating_detected = False
        
        # Restore auto-advance state (typically we want to continue auto-advancing after reset)
        st.session_state.auto_advance = was_auto_advancing
    
    def advance(self):
        """
        Advance the simulation one step.
        
        Returns:
            bool: True if advanced successfully, False if stopped
        """
        if st.session_state.step < self.config.num_steps - 1 and not st.session_state.paused:
            # Store the current grid state before updating
            current_grid = st.session_state.grid.copy()
            
            # Get the rule function from the app
            rule_fn = self.app.get_rule_function()
            
            # Update the grid based on the rule
            new_grid = rule_fn(current_grid)
            st.session_state.grid = new_grid
            st.session_state.step += 1
            
            # Check for stable pattern (no change between steps)
            if np.array_equal(current_grid, new_grid):
                st.session_state.stable_detected = True
                return False  # Stop auto-advance
            
            # Add current state to history for repeating pattern detection
            grid_hash = hash(new_grid.tobytes())
            st.session_state.pattern_history.append(grid_hash)
            
            # Keep history at a manageable size
            if len(st.session_state.pattern_history) > st.session_state.history_length:
                st.session_state.pattern_history.pop(0)
            
            # Check for repeating patterns
            # We need at least 4 states to detect a meaningful pattern (2 complete cycles)
            if len(st.session_state.pattern_history) >= 4:
                # Look for repeating subsequences in the history
                history = st.session_state.pattern_history
                for period in range(1, len(history) // 2):
                    # Check if the last 'period' states match the previous 'period' states
                    if history[-period:] == history[-2*period:-period]:
                        st.session_state.repeating_detected = True
                        return False  # Stop auto-advance
            
            return True
        return False
    
    def toggle_pause(self):
        """Toggle the pause state"""
        st.session_state.paused = not st.session_state.paused
        st.session_state.auto_advance = not st.session_state.paused
    
    def render_controls(self):
        """Render the animation control buttons and speed controls"""
        # Improved control layout with more intuitive options
        col1, col2, col3, col4 = st.columns(4)
        
        # Reset button
        with col1:
            if st.button('↺ Reset'):
                self.reset()
                # Auto-start after reset for better UX
                st.session_state.paused = False
                st.session_state.auto_advance = True
        
        # Play/Pause button
        with col2:
            if st.session_state.paused:
                if st.button('▶️ Play'):
                    self.toggle_pause()
            else:
                if st.button('⏸️ Pause'):
                    self.toggle_pause()
        
        # Step button
        with col3:
            if st.button('⏭️ Step'):
                # Allow manual stepping even when paused
                temp_paused = st.session_state.paused
                st.session_state.paused = False
                self.advance()
                st.session_state.paused = temp_paused
        
        # Speed control
        with col4:
            # Add a speed multiplier for more control
            speed_options = {
                'Slow': 1.5,
                'Normal': 1.0,
                'Fast': 0.5,
                'Very Fast': 0.2
            }
            selected_speed = st.selectbox(
                'Speed',
                options=list(speed_options.keys()),
                index=1  # Default to 'Normal'
            )
            
            # Store the speed multiplier in session state
            st.session_state.speed_multiplier = speed_options[selected_speed]
        
        # Add a progress bar to show simulation progress
        if st.session_state.step > 0:
            progress = st.session_state.step / self.config.num_steps
            st.progress(progress)
    
    def render_grid(self):
        """Render the current grid state"""
        # Display current step
        st.subheader(f"Step {st.session_state.step + 1} of {self.config.num_steps}")
        
        # Create and display the grid visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(st.session_state.grid, cmap=config.cmap, interpolation='nearest')
        ax.axis('off')
        st.pyplot(fig)
    
    def render_status_messages(self):
        """Render status messages based on current state"""
        # Create a status area with consistent height
        status_container = st.container()
        
        with status_container:
            # Show current simulation status
            if st.session_state.auto_advance and not st.session_state.paused:
                st.info("📽️ Simulation running in real-time...")
            elif st.session_state.paused:
                st.warning("⏸️ Simulation paused. Press 'Play' to continue or 'Step' for single step.")
                
            # Show completion message when done
            if st.session_state.step >= self.config.num_steps - 1:
                st.success("🎉 Simulation complete! Click 'Reset' to start over.")
            
            # Show pattern detection notifications
            if st.session_state.stable_detected:
                st.info("🔄 Stable pattern detected! The grid is no longer changing between steps.")
                st.info("This is a common end state in cellular automata. Try 'Reset' with a different initial pattern.")
            
            if st.session_state.repeating_detected:
                st.info("🔁 Repeating pattern detected! The grid is cycling through the same states.")
                st.info("This is a common behavior in cellular automata. Try 'Reset' with a different initial pattern or rule set.")
    
    def update(self):
        """Update the animation state if auto-advancing"""
        if st.session_state.auto_advance:
            # Apply speed multiplier to the delay
            if 'speed_multiplier' not in st.session_state:
                st.session_state.speed_multiplier = 1.0  # Default if not set
                
            # Calculate actual delay based on speed multiplier
            actual_delay = self.config.delay * st.session_state.speed_multiplier
            
            # Only sleep if we're auto-advancing
            time.sleep(actual_delay)
            
            # Turn off auto-advance if we've reached the end
            if not self.advance():
                st.session_state.auto_advance = False
            st.rerun()

# Create animation state manager
animation = AnimationState(app, config)

# Render UI controls
animation.render_controls()

# Render current grid state
animation.render_grid()

# Render status messages
animation.render_status_messages()

# Update animation state
animation.update()