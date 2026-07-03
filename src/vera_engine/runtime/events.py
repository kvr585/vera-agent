"""Lightweight synchronous event dispatcher for decoupling engine components."""

from collections.abc import Callable

from vera_engine.core.entities import Event


class EventDispatcher:
    """Synchronous, in-memory event bus implementing the pub/sub pattern."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[[Event], None]]] = {}
        self._global_listeners: list[Callable[[Event], None]] = []

    def subscribe(self, event_name: str, listener: Callable[[Event], None]) -> None:
        """Subscribes a listener callback to a specific event name.

        Args:
            event_name: The name of the event to listen for.
            listener: A callable that receives the Event object.
        """
        if event_name == "*":
            self._global_listeners.append(listener)
            return

        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(listener)

    def dispatch(self, event: Event) -> None:
        """Dispatches an event, invoking all registered listener callbacks.

        Args:
            event: The Event instance containing name, payload, and timestamp.
        """
        # Invoke global wildcard listeners
        for listener in self._global_listeners:
            try:
                listener(event)
            except Exception:
                # In production, we don't want a listener crash to halt the kernel.
                # Since logging is set up later, we suppress or let it handle carefully.
                pass

        # Invoke event-specific listeners
        if event.name in self._listeners:
            for listener in self._listeners[event.name]:
                try:
                    listener(event)
                except Exception:
                    pass
