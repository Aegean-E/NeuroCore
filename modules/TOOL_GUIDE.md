# 🛠️ NeuroCore Tool Library Guide

> **Empower your AI agents with custom Python functions through OpenAI-compatible Function Calling**

---

## 📑 Table of Contents

1. [What is a Tool?](#1-what-is-a-tool)
2. [Architecture Overview](#2-architecture-overview)
3. [Built-in Tools Reference](#3-built-in-tools-reference)
4. [Creating Custom Tools](#4-creating-custom-tools)
5. [Writing Python Logic](#5-writing-python-logic)
6. [Using Tools in AI Flows](#6-using-tools-in-ai-flows)
7. [Advanced Features](#7-advanced-features)
8. [Importing and Exporting](#8-importing-and-exporting)
9. [Best Practices](#9-best-practices)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What is a Tool?

A **Tool** is a bridge between your AI agent and the external world. It consists of two essential parts:

| Component | Purpose | Format |
|-----------|---------|--------|
| **Definition** | Describes the function to the LLM so it knows *how* and *when* to call it | JSON Schema |
| **Implementation** | The actual Python code that executes when the tool is called | Python Code |

### Tool Execution Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│  LLM Core   │────▶│  Tool Call  │────▶│  Tool       │
│   Request   │     │  (decides)  │     │  Request    │     │  Dispatcher │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                     │
                              ┌────────────────────────────────────┘
                              ▼
                       ┌─────────────┐
                       │  Python     │
                       │  Execution  │
                       └──────┬──────┘
                              │
                              ▼
                       ┌─────────────┐     ┌─────────────┐
                       │  Result     │────▶│  LLM        │
                       │  Return     │     │  Response   │
                       └─────────────┘     └─────────────┘
```

---

## 2. Architecture Overview

The Tool Library is built on these core components:

- **🔧 Tool Definitions** (`tools.json`): Stores tool metadata and JSON schemas
- **📁 Library Directory** (`library/`): Contains Python implementation files
- **⚡ Tool Dispatcher Node**: Executes tool calls within AI Flows
- **🎯 System Prompt Integration**: Injects available tools into LLM context

---

## 3. Built-in Tools Reference

NeuroCore comes with **16 powerful built-in tools** ready to use:

### 🌤️ Information & Utilities

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| **Weather** | Get current weather for any location | `location` (string) |
| **SystemTime** | Get current system date and time | None |
| **TimeZoneConverter** | Convert times between time zones | `source_tz`, `target_tz`, `time_string` |
| **ConversionCalculator** | Convert between units (temp, length, weight, volume) | `value`, `from_unit`, `to_unit` |
| **Calculator** | Evaluate mathematical expressions | `expression` (string) |

### 🌐 Web & Search

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| **FetchURL** | Fetch raw text content from any URL | `url` (string) |
| **WikipediaLookup** | Search and retrieve Wikipedia articles | `query`, `mode` (summary/full) |
| **ArXivSearch** | Search academic papers on ArXiv | `query`, `max_results` |
| **YouTubeTranscript** | Fetch video transcripts | `url` (YouTube URL) |
| **CurrencyConverter** | Real-time currency conversion | `amount`, `from_currency`, `to_currency` |

### 📅 Calendar & Reminders

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| **SaveReminder** | Save calendar events and reminders | `title`, `time` (ISO 8601) |
| **CheckCalendar** | Check calendar for events | `date` (YYYY-MM-DD, optional) |

### 📧 Communication

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| **SendEmail** | Send emails via SMTP | `to_email`, `subject`, `body` |

### 🎯 Goals Management

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| **SetGoal** | Create new goals for the agent | `description`, `priority`, `deadline` |
| **MarkGoalComplete** | Mark goals as completed | `goal_id` (number) |
| **DeleteGoal** | Remove goals from system | `goal_id` (number) |

### Detailed Tool Schemas

#### Weather
```json
{
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "The city and state, e.g. San Francisco, CA"
    }
  },
  "required": ["location"]
}
```

#### Calculator
```json
{
  "type": "object",
  "properties": {
    "expression": {
      "type": "string",
      "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
    }
  },
  "required": ["expression"]
}
```

#### TimeZoneConverter
```json
{
  "type": "object",
  "properties": {
    "time_string": {
      "type": "string",
      "description": "The date and time to convert (ISO 8601 format). If omitted, uses current system time."
    },
    "source_tz": {
      "type": "string",
      "description": "The source time zone (IANA format, e.g., 'America/New_York', 'UTC')"
    },
    "target_tz": {
      "type": "string",
      "description": "The target time zone (IANA format, e.g., 'Europe/London', 'Asia/Tokyo')"
    }
  },
  "required": ["source_tz", "target_tz"]
}
```

#### ConversionCalculator
```json
{
  "type": "object",
  "properties": {
    "value": {
      "type": "number",
      "description": "The numerical value to convert"
    },
    "from_unit": {
      "type": "string",
      "description": "The unit to convert from (e.g., 'meters', 'celsius', 'lbs')"
    },
    "to_unit": {
      "type": "string",
      "description": "The unit to convert to (e.g., 'feet', 'fahrenheit', 'kg')"
    }
  },
  "required": ["value", "from_unit", "to_unit"]
}
```

#### FetchURL
```json
{
  "type": "object",
  "properties": {
    "url": {
      "type": "string",
      "description": "The URL to fetch content from"
    }
  },
  "required": ["url"]
}
```

#### CurrencyConverter
```json
{
  "type": "object",
  "properties": {
    "amount": {
      "type": "number",
      "description": "The amount to convert"
    },
    "from_currency": {
      "type": "string",
      "description": "Source currency code (e.g., USD, EUR, JPY)"
    },
    "to_currency": {
      "type": "string",
      "description": "Target currency code (e.g., EUR, GBP, CAD)"
    }
  },
  "required": ["amount", "from_currency", "to_currency"]
}
```

#### WikipediaLookup
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The search term or topic"
    },
    "mode": {
      "type": "string",
      "enum": ["summary", "full"],
      "description": "Whether to retrieve just the summary or full article text"
    }
  },
  "required": ["query"]
}
```

#### ArXivSearch
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The search query (keywords, topics, paper titles)"
    },
    "max_results": {
      "type": "number",
      "description": "Maximum number of results to return (default 5, max 20)"
    }
  },
  "required": ["query"]
}
```

#### YouTubeTranscript
```json
{
  "type": "object",
  "properties": {
    "url": {
      "type": "string",
      "description": "The URL of the YouTube video"
    }
  },
  "required": ["url"]
}
```

#### SaveReminder
```json
{
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "description": "The title of the event"
    },
    "time": {
      "type": "string",
      "description": "The date and time (ISO 8601 preferred). If omitted, uses current time"
    }
  },
  "required": ["title"]
}
```

#### CheckCalendar
```json
{
  "type": "object",
  "properties": {
    "date": {
      "type": "string",
      "description": "The date to check (YYYY-MM-DD). If omitted, returns upcoming events"
    }
  },
  "required": []
}
```

#### SendEmail
```json
{
  "type": "object",
  "properties": {
    "to_email": {
      "type": "string",
      "description": "The recipient's email address"
    },
    "subject": {
      "type": "string",
      "description": "The subject of the email"
    },
    "body": {
      "type": "string",
      "description": "The content of the email"
    }
  },
  "required": ["to_email", "subject", "body"]
}
```

#### SetGoal
```json
{
  "type": "object",
  "properties": {
    "description": {
      "type": "string",
      "description": "The goal description - what should be achieved"
    },
    "priority": {
      "type": "number",
      "description": "Priority level (higher = more important, default 0)"
    },
    "deadline": {
      "type": "number",
      "description": "Optional deadline as Unix timestamp"
    }
  },
  "required": ["description"]
}
```

#### MarkGoalComplete / DeleteGoal
```json
{
  "type": "object",
  "properties": {
    "goal_id": {
      "type": "number",
      "description": "The ID of the goal"
    }
  },
  "required": ["goal_id"]
}
```

---

## 4. Creating Custom Tools

Navigate to **Settings** -> **Tools** in the NeuroCore dashboard. Click **+ New Tool**.

### Required Fields

| Field | Description | Tips |
|-------|-------------|------|
| **Function Name** | Unique identifier (snake_case) | Use descriptive names like `get_weather`, `search_database` |
| **Description** | Clear explanation for the LLM | Be specific: "Calculates square root" not "Math function" |
| **Parameters** | JSON Schema object | Define types, descriptions, and required fields |
| **Python Logic** | Implementation code | Use `args` for input, `result` for output |

### Parameter Schema Example

```json
{
  "type": "object",
  "properties": {
    "location": { 
      "type": "string", 
      "description": "City and state, e.g. San Francisco, CA" 
    },
    "unit": { 
      "type": "string", 
      "enum": ["celsius", "fahrenheit"],
      "description": "Temperature unit"
    }
  },
  "required": ["location"]
}
```

---

## 5. Writing Python Logic

The Python environment executes locally with access to standard libraries and installed packages.

### Available Variables

| Variable | Type | Description |
|----------|------|-------------|
| `args` | `dict` | Arguments provided by the LLM |
| `result` | `any` | Assign your return value here |
| `json` | `module` | Pre-imported JSON module |
| `httpx` | `module` | HTTP client for API calls |
| `asyncio` | `module` | Async support |

### Complete Example: Weather Tool

**Parameters:**
```json
{
  "type": "object",
  "properties": {
    "location": { "type": "string" }
  },
  "required": ["location"]
}
```

**Python Code:**
```python
import httpx  # Ensure httpx is installed

location = args.get('location')

if not location:
    result = "Error: No location provided."
else:
    try:
        # Example API call
        resp = httpx.get(f"https://wttr.in/{location}?format=3", timeout=5)
        result = resp.text.strip()
    except Exception as e:
        result = f"Failed to fetch weather: {e}"
```

### Advanced Python Patterns

#### Handling Optional Parameters
```python
# Provide defaults for optional args
limit = args.get('limit', 10)
offset = args.get('offset', 0)
```

#### Type Conversion
```python
# Ensure proper types
count = int(args.get('count', 0))
enabled = bool(args.get('enabled', False))
```

#### Working with Lists
```python
items = args.get('items', [])
if not isinstance(items, list):
    items = [items]  # Normalize single item to list
```

---

## 6. Using Tools in AI Flows

To enable tool usage, configure the **System Prompt** and add a **Tool Dispatcher**.

### Step 1: Enable Tools in System Prompt

1. Open your AI Flow
2. Select the **System Prompt** node
3. In the configuration panel, check the tools you want to enable
4. This injects tool definitions into the LLM context

### Step 2: Add the Tool Dispatcher Node

The LLM requests tool calls, but cannot execute code itself. The **Tool Dispatcher** handles execution.

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  System     │────▶│     LLM Core     │────▶│ Conditional │
│  Prompt     │     │  (with tools)    │     │   Router    │
│  (+tools)   │     └──────────────────┘     └──────┬──────┘
└─────────────┘                                     │
                              ┌──────────────────────┘
                              │ check_field: "tool_calls"
                              ▼
                       ┌─────────────┐     ┌─────────────┐
                       │   Tool      │────▶│    LLM      │
                       │  Dispatcher │     │   (interpret)│
                       └─────────────┘     └─────────────┘
```

**Configuration Steps:**

1. Add a **Tool Dispatcher** node to your flow
2. Connect **LLM Core** output to **Tool Dispatcher** input
3. Use a **Conditional Router** to check if `tool_calls` exists
4. Connect **Tool Dispatcher** output back to **LLM Core** for result interpretation

### Tool Dispatcher Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `allowed_tools` | List of permitted tools (empty = all) | `[]` |
| `max_tools_per_turn` | Maximum tools per execution | `5` |

---

## 7. Advanced Features

### Tool Configuration

Some tools support additional configuration:

#### SendEmail SMTP Settings
Configure via environment variables or the tool config UI:
- `SMTP_SERVER` - SMTP server hostname
- `SMTP_PORT` - Server port (default: 587)
- `SMTP_EMAIL` - Sender email address
- `SMTP_PASSWORD` - Authentication password

### Error Handling Strategies

The Tool Dispatcher provides robust error handling:

| Error Type | Behavior |
|------------|----------|
| JSON Parse Error | Returns error message to LLM |
| Tool Not Found | Returns "Tool not found" error |
| Execution Error | Returns exception details |
| Tool Not Allowed | Returns "Tool not enabled" error |

### Rate Limiting

Control tool execution with `max_tools_per_turn`:
- Prevents runaway tool loops
- Remaining tools are queued for next turn
- Tracked via `_tool_count` and `_remaining_tool_calls`

---

## 8. Importing and Exporting

Share tools between NeuroCore instances or back them up.

### Exporting Tools

| Method | Action | Output |
|--------|--------|--------|
| Single Tool | Click **Export Tool** | `{name}.json` |
| Entire Library | Click **Export Entire Library** | `neurocore_tools.json` |

**Export Format:**
```json
[
  {
    "name": "Weather",
    "enabled": true,
    "description": "Get current weather...",
    "parameters": { ... },
    "code": "import httpx\n..."
  }
]
```

### Importing Tools

| File Type | Handling |
|-----------|----------|
| `.json` | Full tool definition with code |
| `.py` | Wrapped as tool (adjust schema manually) |

**Import Steps:**
1. Click **Import Tool**
2. Select `.json` or `.py` file
3. Review and adjust parameter schema if needed
4. Save

---

## 9. Best Practices

### Writing Good Tool Descriptions

✅ **Good Examples:**
- "Fetches current weather conditions including temperature, humidity, and forecast"
- "Searches the product database by name and returns matching items with price and availability"
- "Converts between 50+ currencies using real-time exchange rates"

❌ **Bad Examples:**
- "Weather tool"
- "Database search"
- "Money converter"

### Parameter Design

1. **Use clear, descriptive names** - `location` not `loc`
2. **Provide examples in descriptions** - "e.g., San Francisco, CA"
3. **Mark truly required fields** - Don't overuse `required`
4. **Use enums for limited options** - `["celsius", "fahrenheit"]`
5. **Set sensible defaults** - Handle missing optional parameters gracefully

### Security Considerations

- Use `allowed_tools` in Tool Dispatcher to restrict access
- Validate all inputs before processing
- Never expose sensitive credentials in tool code
- Use environment variables for API keys
- Sanitize user inputs to prevent injection attacks

### Performance Tips

- Keep tool execution fast (< 5 seconds ideal)
- Use timeouts for external API calls
- Cache results when appropriate
- Handle rate limits gracefully

---

## 10. Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| LLM doesn't call tools | Tools not enabled in System Prompt | Check tool checkboxes in System Prompt config |
| "Tool not found" error | Tool name mismatch | Verify function name matches definition |
| JSON parse errors | Malformed LLM arguments | Tool Dispatcher handles this automatically |
| Tool executes but no result | Missing `result` assignment | Ensure `result = ...` in Python code |
| Import fails | Invalid JSON or encoding | Validate JSON syntax, check file encoding |

### Debugging Steps

1. **Check Tool Definitions**
   ```bash
   cat modules/tools/tools.json | jq '.'
   ```

2. **Verify Python Code**
   - Check syntax with `python -m py_compile library/{tool_name}.py`
   - Ensure `result` variable is assigned

3. **Test in Isolation**
   - Create a simple flow with just System Prompt → LLM Core → Tool Dispatcher
   - Enable only one tool at a time

4. **Review Logs**
   - Check console output for execution errors
   - Look for `Error executing tool` messages

### Getting Help

- Check the [NeuroCore Documentation](https://docs.neurocore.ai)
- Review example tools in `modules/tools/library/`
- Enable debug mode in settings for verbose logging

---

> 💡 **Pro Tip:** Start with built-in tools to understand patterns, then create custom tools for your specific use cases!
