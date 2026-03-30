import sys
import threading
import time
from typing import Optional

class FakeProgressBar:
    """A threaded progress bar that smoothly increments while a blocking task runs.
    
    This is used to simulate tqdm progress bars during JAX block_until_ready() calls
    without interrupting the GPU execution.
    """
    def __init__(self, desc: str, total_time: float = 10.0, width: int = 40):
        self.desc = desc
        self.total_time = total_time
        self.width = width
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def _run(self):
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            # Increment to 99%, leave the last 1% for completion
            progress = min(0.99, elapsed / self.total_time)
            self._print_bar(progress)
            time.sleep(0.1)

    def _print_bar(self, progress: float):
        filled = int(self.width * progress)
        empty = self.width - filled
        bar = "#" * filled + " " * empty
        pct = int(progress * 100)
        sys.stdout.write(f"\r{self.desc}: {pct:>3}%|{bar}|")
        sys.stdout.flush()

    def start(self):
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        # Instantly fill to 100%
        self._print_bar(1.0)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
