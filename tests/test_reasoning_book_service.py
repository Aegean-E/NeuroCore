import pytest
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.reasoning_book.service import ReasoningBookService, MAX_THOUGHTS


@pytest.fixture
def temp_data_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def service(temp_data_file, monkeypatch):
    monkeypatch.setattr("modules.reasoning_book.service.DATA_FILE", temp_data_file)
    return ReasoningBookService()


def test_log_thought(service):
    service.log_thought("Test thought", source="Test")
    
    thoughts = service.get_thoughts()
    assert len(thoughts) == 1
    assert thoughts[0]["content"] == "Test thought"
    assert thoughts[0]["source"] == "Test"
    assert "timestamp" in thoughts[0]


def test_log_thought_default_source(service):
    service.log_thought("Another thought")
    
    thoughts = service.get_thoughts()
    assert len(thoughts) == 1
    assert thoughts[0]["source"] == "Flow"


def test_get_thoughts(service):
    service.log_thought("Thought 1")
    service.log_thought("Thought 2")
    
    thoughts = service.get_thoughts()
    
    assert len(thoughts) == 2
    assert thoughts[0]["content"] == "Thought 2"
    assert thoughts[1]["content"] == "Thought 1"


def test_clear(service):
    service.log_thought("Thought 1")
    service.log_thought("Thought 2")
    
    service.clear()
    
    thoughts = service.get_thoughts()
    assert len(thoughts) == 0


def test_max_thoughts_limit(service):
    for i in range(MAX_THOUGHTS + 10):
        service.log_thought(f"Thought {i}")
    
    thoughts = service.get_thoughts()
    assert len(thoughts) == MAX_THOUGHTS
