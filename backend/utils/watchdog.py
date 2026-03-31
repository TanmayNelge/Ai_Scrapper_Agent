"""
Watchdog monitors pipeline progress and kills stalled operations.
Tracks last_progress_time and a stall counter per branch.
If no progress in N seconds, it signals abort and retry.
"""
import asyncio
import time
from typing import Optional


class OperationTimeout(Exception):
    """Raised when an operation exceeds its time budget."""
    pass


class StallDetected(Exception):
    """Raised when the crawler detects no progress across multiple pages."""
    pass


class Watchdog:
    def __init__(self, stall_timeout: float = 45.0, max_consecutive_stalls: int = 3):
        self.stall_timeout = stall_timeout
        self.max_consecutive_stalls = max_consecutive_stalls
        self._last_progress_time: float = time.time()
        self._consecutive_stalls: int = 0
        self._total_items_extracted: int = 0
        self._pages_processed: int = 0

    def record_progress(self):
        """Call this whenever we extract data or discover new useful links."""
        self._last_progress_time = time.time()
        self._consecutive_stalls = 0

    def record_extraction(self, count: int = 1):
        """Call when items are successfully extracted."""
        self._total_items_extracted += count
        self.record_progress()

    def record_page_processed(self):
        self._pages_processed += 1

    def record_stall(self):
        """Call when a page yields nothing useful."""
        self._consecutive_stalls += 1

    def is_stalled(self) -> bool:
        """Check if we've exceeded the stall threshold."""
        time_stalled = time.time() - self._last_progress_time
        return (
            time_stalled > self.stall_timeout
            or self._consecutive_stalls >= self.max_consecutive_stalls
        )

    def should_abandon_branch(self) -> bool:
        """True if we should give up on the current branch and try next seed."""
        return self._consecutive_stalls >= self.max_consecutive_stalls

    def reset_for_new_branch(self):
        """Reset stall counters when switching to a new seed/branch."""
        self._consecutive_stalls = 0
        self._last_progress_time = time.time()

    @property
    def stats(self) -> dict:
        return {
            "total_items": self._total_items_extracted,
            "pages_processed": self._pages_processed,
            "consecutive_stalls": self._consecutive_stalls,
            "seconds_since_progress": round(time.time() - self._last_progress_time, 1),
        }


async def run_with_timeout(coro, timeout: float, fallback=None, label: str = "operation"):
    """
    Execute a coroutine with a hard timeout.
    Returns fallback value if timeout or error occurs.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Watchdog] {label} timed out after {timeout}s")
        return fallback
    except Exception as e:
        print(f"[Watchdog] {label} failed: {e}")
        return fallback


async def retry_async(coro_factory, max_retries: int = 3, delay: float = 1.0,
                      timeout: float = 60.0, label: str = "operation"):
    """
    Retry a coroutine factory with exponential backoff.
    coro_factory must be a callable that returns a fresh coroutine each time.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = await asyncio.wait_for(coro_factory(), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            last_error = f"Timeout after {timeout}s"
            print(f"[Retry] {label} attempt {attempt}/{max_retries}: {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"[Retry] {label} attempt {attempt}/{max_retries}: {last_error}")

        if attempt < max_retries:
            wait = delay * (2 ** (attempt - 1))  # Exponential backoff
            await asyncio.sleep(wait)

    print(f"[Retry] {label} exhausted all {max_retries} retries. Last error: {last_error}")
    return None
