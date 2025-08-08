import logging
import sys
import signal
import atexit

# A lot of the details in the functions below were GPT-generated since I keep forgetting the syntax details.
def flush_all_logs():
    """Force flush all logging handlers"""
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    
def signal_handler(signum, frame):
    """Handle signals (like Ctrl+C) by flushing logs before exit"""
    flush_all_logs()
    sys.exit(0)

def setup_logging():
    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True
    )
    
    signal.signal(signal.SIGINT, signal_handler)  
    signal.signal(signal.SIGTERM, signal_handler)
    
    atexit.register(flush_all_logs)
    
    for handler in logging.getLogger().handlers:
        handler.flush = lambda: handler.stream.flush() if hasattr(handler, 'stream') else None

def get_logger(name: str):
    logger = logging.getLogger(name)
    
    # Create a custom log method that flushes immediately
    original_log = logger._log
    
    def log_with_flush(level, msg, args, **kwargs):
        original_log(level, msg, args, **kwargs)
        # Force flush all handlers
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        # Also flush root logger handlers
        for handler in logging.getLogger().handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
    
    logger._log = log_with_flush
    return logger