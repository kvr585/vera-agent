"""Unit tests for the EventDispatcher."""

from vera_engine.core.entities import Event
from vera_engine.runtime.events import EventDispatcher


def test_specific_event_subscription() -> None:
    """Verifies that a listener only receives events it subscribed to."""
    dispatcher = EventDispatcher()
    received_events: list[Event] = []

    def listener(event: Event) -> None:
        received_events.append(event)

    dispatcher.subscribe("GoalReceived", listener)

    # Dispatch matching event
    event_1 = Event(name="GoalReceived", payload={"goal": "Test VERA"})
    dispatcher.dispatch(event_1)

    # Dispatch non-matching event
    event_2 = Event(name="TaskStarted", payload={"task_id": "1"})
    dispatcher.dispatch(event_2)

    assert len(received_events) == 1
    assert received_events[0].name == "GoalReceived"
    assert received_events[0].payload["goal"] == "Test VERA"


def test_global_wildcard_subscription() -> None:
    """Verifies that wildcard '*' listeners receive every dispatched event."""
    dispatcher = EventDispatcher()
    received_events: list[Event] = []

    def global_listener(event: Event) -> None:
        received_events.append(event)

    dispatcher.subscribe("*", global_listener)

    event_1 = Event(name="GoalReceived", payload={"goal": "Test VERA"})
    event_2 = Event(name="TaskStarted", payload={"task_id": "1"})

    dispatcher.dispatch(event_1)
    dispatcher.dispatch(event_2)

    assert len(received_events) == 2
    assert received_events[0].name == "GoalReceived"
    assert received_events[1].name == "TaskStarted"


def test_listener_exception_isolation() -> None:
    """Verifies that a failing listener callback does not disrupt other listeners.

    A failing listener should not raise an error up to the dispatcher.
    """
    dispatcher = EventDispatcher()
    called_second_listener = False

    def buggy_listener(event: Event) -> None:
        raise ValueError("Simulated listener bug")

    def stable_listener(event: Event) -> None:
        nonlocal called_second_listener
        called_second_listener = True

    dispatcher.subscribe("TestEvent", buggy_listener)
    dispatcher.subscribe("TestEvent", stable_listener)

    event = Event(name="TestEvent")

    # Should not raise exception
    dispatcher.dispatch(event)

    assert called_second_listener is True


def test_global_listener_exception_isolation() -> None:
    """Verifies that a failing global listener does not block other executions."""
    dispatcher = EventDispatcher()
    called_second_listener = False

    def buggy_global(event: Event) -> None:
        raise ValueError("Buggy global listener")

    def stable_listener(event: Event) -> None:
        nonlocal called_second_listener
        called_second_listener = True

    dispatcher.subscribe("*", buggy_global)
    dispatcher.subscribe("TestEvent", stable_listener)

    dispatcher.dispatch(Event(name="TestEvent"))

    assert called_second_listener is True
