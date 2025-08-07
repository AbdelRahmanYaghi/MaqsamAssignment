import logging

def setup_logging():
    logging.basicConfig(
        filename="app.log",
        encoding="utf-8",
        # The two lines below were GPT generated
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def get_logger(name: str):
    return logging.getLogger(name)