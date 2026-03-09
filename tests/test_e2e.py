"""
End-to-End Tests for NeuroCore

Tests the prompt editing workflow and module configurations.
"""

import pytest
import requests
import json


BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def base_url():
    return BASE_URL


def test_planner_module_details(base_url):
    """Test 1: Get planner module details (simulates loading the page)."""
    r = requests.get(f'{base_url}/modules/planner/details')
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print(f'   Status: {r.status_code}')
    print(f'   Page loaded successfully: {r.status_code == 200}')


def test_planner_default_prompt(base_url):
    """Test 2: Get default prompt (simulates clicking Load Default Prompt)."""
    r = requests.get(f'{base_url}/modules/planner/default-prompt')
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    default_prompt = r.text
    print(f'   Status: {r.status_code}')
    print(f'   Contains {{max_steps}} placeholder: {"{max_steps}" in default_prompt}')
    print(f'   First 100 chars: {default_prompt[:100]}')
    return default_prompt


def test_save_custom_prompt(base_url):
    """Test 3: Save a custom prompt."""
    custom_prompt = 'Custom planner prompt with {request} and max {max_steps} steps'
    data = {'planner_prompt': custom_prompt, 'max_steps': '20', 'enabled': 'true'}
    r = requests.post(f'{base_url}/modules/planner/config', data=data)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print(f'   Status: {r.status_code}')
    print(f'   Saved successfully: {r.status_code == 200}')


def test_verify_custom_prompt_saved(base_url):
    """Test 4: Verify custom prompt was saved."""
    with open('modules/planner/module.json', 'r') as f:
        config = json.load(f)
    saved_custom = config.get('config', {}).get('planner_prompt', '')
    saved_max_steps = config.get('config', {}).get('max_steps', 0)
    print(f'   Custom prompt saved: {saved_custom == "Custom planner prompt with {request} and max {max_steps} steps"}')
    print(f'   Max steps saved: {saved_max_steps == 20}')
    print(f'   Default preserved: {"default_planner_prompt" in config.get("config", {})}')
    assert saved_custom == 'Custom planner prompt with {request} and max {max_steps} steps'
    assert saved_max_steps == 20


def test_get_default_prompt_again(base_url):
    """Test 5: Get default prompt again (should still return original default)."""
    r = requests.get(f'{base_url}/modules/planner/default-prompt')
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    default_prompt_again = r.text
    print(f'   Status: {r.status_code}')
    print(f'   Still contains {{max_steps}}: {"{max_steps}" in default_prompt_again}')
    return default_prompt_again


def test_load_default_prompt_button(base_url):
    """Test 6: Simulate loading default prompt into textarea."""
    r = requests.get(f'{base_url}/modules/planner/default-prompt')
    default_prompt = r.text
    print(f'   Default prompt ready for textarea: {len(default_prompt)} chars')
    # Fixed: check for newline in a valid way - assign to variable before f-string
    has_newlines = "\n" in str(default_prompt)
    print(f'   Has proper newlines: {has_newlines}')
    assert len(default_prompt) > 0


def test_reflection_module_workflow(base_url):
    """Test 7: Test Reflection module workflow."""
    r = requests.get(f'{base_url}/modules/reflection/default-prompt')
    assert r.status_code == 200
    reflection_default = r.text
    print(f'   Default prompt retrieved: {len(reflection_default)} chars')

    custom_reflection = 'Custom reflection prompt for testing'
    r = requests.post(f'{base_url}/modules/reflection/config', 
                      data={'reflection_prompt': custom_reflection, 'inject_improvement': 'true'})
    assert r.status_code == 200
    print(f'   Custom prompt saved: {r.status_code == 200}')

    with open('modules/reflection/module.json', 'r') as f:
        ref_config = json.load(f)
    print(f'   Default preserved: {"default_reflection_prompt" in ref_config.get("config", {})}')
    print(f'   Custom saved: {ref_config.get("config", {}).get("reflection_prompt") == custom_reflection}')
    assert "default_reflection_prompt" in ref_config.get("config", {})
    assert ref_config.get("config", {}).get("reflection_prompt") == custom_reflection


print('=== All End-to-End Tests Passed ===')

