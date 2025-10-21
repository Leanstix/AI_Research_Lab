# app/__init__.py
from dotenv import load_dotenv, find_dotenv

# Load the first .env found (typically at your project root).
# override=False so real env vars still win if you exported them.
load_dotenv(find_dotenv(), override=False)
