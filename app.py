"""
Main application entry point for the AI CV Ranking System.
"""
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.streamlit_app import StreamlitApp

def main():
    """Main application entry point."""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
