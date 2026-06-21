# Run Server

To run the server, open the terminal inside the project directory and run:

```bash id="kg20cl"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

This command starts the server locally with auto-reload enabled, so changes will be reflected automatically during development.
