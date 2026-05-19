import asyncio


class LockService:
    """
    Service responsible for managing in-memory asynchronous locks on a per-user basis.

    Provides a thread-safe-like synchronization mechanism for async tasks,
    ensuring a single user can only execute one heavy operation sequence
    at any given time.
    """

    def __init__(self):
        """Initializes the LockService with an empty in-memory lock registry."""
        # Maps user_id (int) to its corresponding asyncio.Lock instance
        self._locks = {}

    def get_lock(self, user_id: int) -> asyncio.Lock:
        """
        Retrieves an existing lock for a specific user, or creates a new one
        if it does not exist in the registry.

        This allows handlers to implement user-isolated critical sections.

        Args:
            user_id (int): The unique Telegram ID of the user.

        Returns:
            asyncio.Lock: The asynchronous lock instance tied to the user.
        """
        if user_id not in self._locks:
            # Instantiate a fresh primitive lock for this specific user footprint
            self._locks[user_id] = asyncio.Lock()

        return self._locks[user_id]
