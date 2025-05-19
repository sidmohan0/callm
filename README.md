# CallM: AI-Powered Cellular Automata Explorer

CallM is a streamlined, interactive application for exploring and experimenting with cellular automata. Built with Python and Streamlit, it offers a seamless way to visualize complex emergent behaviors from simple rule sets.

## Features

- **Real-time Simulation**: Watch cellular patterns evolve with adjustable speed controls
- **Multiple Rule Sets**: Explore classic cellular automata including Brian's Brain, Seeds, and Noise-Life
- **Pattern Detection**: Automatic identification of stable and repeating patterns
- **Intuitive Controls**: Play, pause, step, and reset with a clean, responsive interface

## AI-Powered Rule Generation

CallM introduces a novel approach to cellular automata experimentation: **natural language rule creation**. This groundbreaking feature allows you to:

- **Describe rules in plain English**: Simply explain how you want cells to behave
- **Instant code generation**: The application leverages OpenAI's language models to translate your description into functional Python code
- **No programming required**: Create complex cellular automata without writing a single line of code
- **Immediate visualization**: See your custom rules in action instantly

This bridges the gap between creative concept and technical implementation, enabling artists, scientists, educators, and hobbyists to experiment with cellular automata regardless of their programming background.

## Getting Started

### Prerequisites

- Python 3.7+
- Streamlit
- NumPy
- Matplotlib
- OpenAI API key (optional, for custom rule generation)

### Installation

```bash
# Clone the repository
git clone https://github.com/sidmohan0/callm.git
cd callm

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run callm/app.py
```

### Configuration

To use the custom rule generation feature, you'll need an OpenAI API key:

1. Create a `.streamlit/secrets.toml` file in the project root
2. Add your API key: `OPENAI_API_KEY = "your-api-key-here"`

## How It Works

Cellular automata are discrete models where cells on a grid evolve through a series of time steps according to fixed rules. Each cell's state depends on its previous state and the states of neighboring cells.

CallM implements several classic rule sets:

- **Brian's Brain**: A three-state cellular automaton with alive, dying, and dead states
- **Seeds**: A binary cellular automaton where cells are born with exactly 2 neighbors
- **Noise-Life**: A variant of Conway's Game of Life with random noise
- **Custom Rules**: Create your own rules using natural language descriptions

## Contributing

Contributions are welcome! Here are some ways you can contribute:

- Add new cellular automaton rule sets
- Improve performance for large grid sizes
- Enhance the UI with additional visualization options
- Add state persistence to save and load interesting patterns

Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the work of John Conway, Brian Silverman, and other cellular automata pioneers
- Built with Streamlit's interactive framework
- Custom rule generation powered by OpenAI's language models
