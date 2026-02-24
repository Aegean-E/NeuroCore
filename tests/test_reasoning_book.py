import pytest
import os
import json
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock


@pytest.fixture
def temp_reasoning_file():
    """Create a temporary reasoning file for testing."""
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def reasoning_service(temp_reasoning_file):
    """Create a ReasoningBookService with temporary storage."""
    with patch('modules.reasoning_book.service.DATA_FILE', temp_reasoning_file):
        from modules.reasoning_book.service import ReasoningBookService
        service = ReasoningBookService()
        yield service


def test_log_thought(reasoning_service):
    """Test logging a thought."""
    reasoning_service.log_thought("Test thought content", source="Test")
    
    thoughts = reasoning_service.get_thoughts()
    assert len(thoughts) == 1
    assert thoughts[0]['content'] == "Test thought content"
    assert thoughts[0]['source'] == "Test"


def test_log_thought_default_source(reasoning_service):
    """Test logging with default source."""
    reasoning_service.log_thought("My thought")
    
    thoughts = reasoning_service.get_thoughts()
    assert len(thoughts) == 1
    assert thoughts[0]['source'] == "Flow"


def test_get_thoughts_empty(reasoning_service):
    """Test getting thoughts when none exist."""
    thoughts = reasoning_service.get_thoughts()
    assert thoughts == []


def test_get_thoughts_multiple(reasoning_service):
    """Test getting multiple thoughts."""
    reasoning_service.log_thought("First thought")
    reasoning_service.log_thought("Second thought")
    reasoning_service.log_thought("Third thought")
    
    thoughts = reasoning_service.get_thoughts()
    assert len(thoughts) == 3


def test_thoughts_ordered_newest_first(reasoning_service):
    """Test that thoughts are ordered newest first."""
    reasoning_service.log_thought("Oldest")
    reasoning_service.log_thought("Middle")
    reasoning_service.log_thought("Newest")
    
    thoughts = reasoning_service.get_thoughts()
    assert thoughts[0]['content'] == "Newest"
    assert thoughts[2]['content'] == "Oldest"


def test_clear_thoughts(reasoning_service):
    """Test clearing all thoughts."""
    reasoning_service.log_thought("Thought 1")
    reasoning_service.log_thought("Thought 2")
    
    reasoning_service.clear()
    
    thoughts = reasoning_service.get_thoughts()
    assert len(thoughts) == 0


def test_thought_has_timestamp(reasoning_service):
    """Test that thoughts have timestamps."""
    reasoning_service.log_thought("Test")
    
    thoughts = reasoning_service.get_thoughts()
    assert 'timestamp' in thoughts[0]


def test_max_thoughts_limit(reasoning_service):
    """Test that thought count is limited to MAX_THOUGHTS."""
    from modules.reasoning_book.service import MAX_THOUGHTS
    
    # Add more than MAX_THOUGHTS
    for i in range(MAX_THOUGHTS + 10):
        reasoning_service.log_thought(f"Thought {i}")
    
    thoughts = reasoning_service.get_thoughts()
    assert len(thoughts) == MAX_THOUGHTS


def test_thought_persistence(reasoning_service):
    """Test that thoughts persist to file."""
    from modules.reasoning_book.service import DATA_FILE
    reasoning_service.log_thought("Persistent thought")
    
    # Read the file directly
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    
    assert len(data) == 1
    assert data[0]['content'] == "Persistent thought"


def test_load_nonexistent_file(reasoning_service):
    """Test loading when file doesn't exist."""
    reasoning_service.clear()
    
    # The file should be empty
    thoughts = reasoning_service.get_thoughts()
    assert thoughts == []


def test_multiple_sources(reasoning_service):
    """Test logging thoughts from different sources."""
    reasoning_service.log_thought("From chat", source="Chat")
    reasoning_service.log_thought("From flow", source="Flow")
    reasoning_service.log_thought("From agent", source="Agent")
    
    thoughts = reasoning_service.get_thoughts()
    sources = [t['source'] for t in thoughts]
    
    assert "Chat" in sources
    assert "Flow" in sources
    assert "Agent" in sources
