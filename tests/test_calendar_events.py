import pytest
import json
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.calendar.events import EventManager


@pytest.fixture
def temp_event_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def event_manager(temp_event_file):
    return EventManager(storage_file=temp_event_file)


def test_add_event(event_manager):
    event = event_manager.add_event("Test Event", "2026-02-25 10:00")
    
    assert event["title"] == "Test Event"
    assert event["start_time"] == "2026-02-25 10:00"
    assert "id" in event
    assert "created_at" in event
    assert event["notified"] is False


def test_get_events_by_date(event_manager):
    event_manager.add_event("Event 1", "2026-02-25 10:00")
    event_manager.add_event("Event 2", "2026-02-25 14:00")
    event_manager.add_event("Event 3", "2026-02-26 09:00")
    
    events = event_manager.get_events_by_date("2026-02-25")
    
    assert len(events) == 2
    assert all(e.get("start_time", "").startswith("2026-02-25") for e in events)


def test_get_upcoming(event_manager):
    future_date = (datetime.now()).strftime("%Y-%m-%d %H:%M")
    past_date = "2020-01-01 10:00"
    
    event_manager.add_event("Future Event", future_date)
    event_manager.add_event("Past Event", past_date)
    
    upcoming = event_manager.get_upcoming(limit=10)
    
    assert len(upcoming) >= 1
    assert all(e.get("start_time", "") >= "2020-01-01 00:00" for e in upcoming)


def test_get_event_by_id(event_manager):
    event = event_manager.add_event("Test Event", "2026-02-25 10:00")
    event_id = event["id"]
    
    found_event = event_manager.get_event_by_id(event_id)
    
    assert found_event is not None
    assert found_event["id"] == event_id
    assert found_event["title"] == "Test Event"


def test_get_event_by_id_not_found(event_manager):
    result = event_manager.get_event_by_id("nonexistent-id")
    
    assert result is None


def test_delete_event(event_manager):
    event = event_manager.add_event("To Delete", "2026-02-25 10:00")
    event_id = event["id"]
    
    event_manager.delete_event(event_id)
    
    found = event_manager.get_event_by_id(event_id)
    assert found is None


def test_delete_nonexistent_event(event_manager):
    event_manager.delete_event("nonexistent-id")
    
    events = event_manager.get_upcoming(limit=10)
    assert len(events) == 0
