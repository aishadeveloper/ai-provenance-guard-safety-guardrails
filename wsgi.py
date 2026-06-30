"""Local entrypoint. Run with: python wsgi.py  (serves on http://localhost:5000)"""

from dotenv import load_dotenv

load_dotenv()  # pull GROQ_API_KEY from .env before the app builds its client

from provenance.app import create_app  # noqa: E402  (must follow load_dotenv)

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
