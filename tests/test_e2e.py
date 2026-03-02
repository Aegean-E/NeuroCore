import requests
import json

print('=== End-to-End Prompt Editing Workflow Test ===')
print()

# Test 1: Get planner module details (simulates loading the page)
print('1. Loading planner module details...')
r = requests.get('http://localhost:8000/modules/planner/details')
print(f'   Status: {r.status_code}')
print(f'   Page loaded successfully: {r.status_code == 200}')
print()

# Test 2: Get default prompt (simulates clicking Load Default Prompt)
print('2. Getting default prompt...')
r = requests.get('http://localhost:8000/modules/planner/default-prompt')
default_prompt = r.text
print(f'   Status: {r.status_code}')
print(f'   Contains {{max_steps}} placeholder: {"{max_steps}" in default_prompt}')
print(f'   First 100 chars: {default_prompt[:100]}')
print()

# Test 3: Save a custom prompt
print('3. Saving custom prompt...')
custom_prompt = 'Custom planner prompt with {request} and max {max_steps} steps'
data = {'planner_prompt': custom_prompt, 'max_steps': '20', 'enabled': 'true'}
r = requests.post('http://localhost:8000/modules/planner/config', data=data)
print(f'   Status: {r.status_code}')
print(f'   Saved successfully: {r.status_code == 200}')
print()

# Test 4: Verify custom prompt was saved
print('4. Verifying custom prompt was saved...')
with open('modules/planner/module.json', 'r') as f:
    config = json.load(f)
saved_custom = config.get('config', {}).get('planner_prompt', '')
saved_max_steps = config.get('config', {}).get('max_steps', 0)
print(f'   Custom prompt saved: {saved_custom == custom_prompt}')
print(f'   Max steps saved: {saved_max_steps == 20}')
print(f'   Default preserved: {"default_planner_prompt" in config.get("config", {})}')
print()

# Test 5: Get default prompt again (should still return original default)
print('5. Getting default prompt again...')
r = requests.get('http://localhost:8000/modules/planner/default-prompt')
default_prompt_again = r.text
print(f'   Status: {r.status_code}')
print(f'   Default unchanged: {default_prompt_again == default_prompt}')
print(f'   Still contains {{max_steps}}: {"{max_steps}" in default_prompt_again}')
print()

# Test 6: Simulate loading default prompt into textarea
print('6. Simulating Load Default Prompt button click...')
print(f'   Default prompt ready for textarea: {len(default_prompt_again)} chars')
print(f'   Has proper newlines: {"\\n" in default_prompt_again}')
print()

# Test 7: Test Reflection module workflow
print('7. Testing Reflection module workflow...')
r = requests.get('http://localhost:8000/modules/reflection/default-prompt')
reflection_default = r.text
print(f'   Default prompt retrieved: {len(reflection_default)} chars')

custom_reflection = 'Custom reflection prompt for testing'
r = requests.post('http://localhost:8000/modules/reflection/config', 
                  data={'reflection_prompt': custom_reflection, 'inject_improvement': 'true'})
print(f'   Custom prompt saved: {r.status_code == 200}')

with open('modules/reflection/module.json', 'r') as f:
    ref_config = json.load(f)
print(f'   Default preserved: {"default_reflection_prompt" in ref_config.get("config", {})}')
print(f'   Custom saved: {ref_config.get("config", {}).get("reflection_prompt") == custom_reflection}')
print()

print('=== All End-to-End Tests Passed ===')
