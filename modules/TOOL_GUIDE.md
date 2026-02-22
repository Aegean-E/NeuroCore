# NeuroCore Tool Library Guide

The **Tool Library** allows you to extend the capabilities of your AI agents by defining custom Python functions that the LLM can execute. This feature implements OpenAI-compatible "Function Calling".

## 1. What is a Tool?

A tool consists of two parts:
1.  **Definition (JSON Schema)**: Describes the function to the LLM so it knows *how* and *when* to call it.
2.  **Implementation (Python Code)**: The actual code that runs when the tool is called.

## 2. Creating a Tool

Navigate to **Settings** -> **Tools** in the NeuroCore dashboard. Click **+ New Tool**.

### Fields

*   **Function Name**: A unique identifier for the tool (e.g., `get_weather`, `calculator`, `search_db`). Use snake_case.

*   **Description**: A clear explanation of what the tool does. The LLM uses this to decide which tool to call.
    *   *Good*: "Calculates the square root of a number."
    *   *Bad*: "Math function."

*   **Parameters (JSON Schema)**: Defines the arguments the tool accepts.
    *   Must be a valid JSON Schema object.
    *   Example:
        ```json
        {
          "type": "object",
          "properties": {
            "location": { "type": "string", "description": "City and state" },
            "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
          },
          "required": ["location"]
        }
        ```

*   **Python Logic**: The code executed when the tool is triggered.

## 3. Writing Python Logic

The Python environment for tools is executed locally. You can import standard libraries or packages installed in your environment.

### Inputs & Outputs
*   **Input**: The arguments provided by the LLM are available in the dictionary variable `args`.

*   **Output**: You must assign your return value to the variable `result`.

*   **Helpers**: The `json` module is pre-imported and available.

*   **Imports**: You can import standard Python libraries (e.g., `math`, `datetime`) or installed packages (e.g., `httpx`) directly in your code.

### Example: Weather Tool

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
import httpx # Ensure httpx is installed in your environment

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

## 4. Using Tools in AI Flows

To make an AI agent use your tools, you need to configure the **System Prompt** and the **Tool Dispatcher**.

### Step 1: Enable Tools in System Prompt
1.  Open your AI Flow.
2.  Select the **System Prompt** node.
3.  In the configuration panel, check the boxes for the tools you want this agent to have access to.

4.  This injects the tool definitions into the LLM context.

### Step 2: Add the Tool Dispatcher Node
The LLM will *request* a tool call, but it cannot execute code itself. The **Tool Dispatcher** node handles execution.

1.  Add a **Tool Dispatcher** node (from the Tools module) to your flow.
2.  Connect the **LLM Core** output to the **Tool Dispatcher** input.

3.  **Important**: You should usually use a **Conditional Router** before the dispatcher to check if the LLM actually requested a tool (check if `tool_calls` exists).
4.  Connect the **Tool Dispatcher** output back to the **LLM Core** input (or a new LLM node) if you want the AI to interpret the tool's result and generate a natural language response.

*Note: The Tool Dispatcher node can also be configured to only allow specific tools, adding a layer of security.*

## 5. Importing and Exporting Tools

You can share tools between NeuroCore instances or back them up using the Import/Export feature.

### Exporting
*   Click **Export Tool** in the **Settings -> Tools** tab.
*   Search for a specific tool to export it individually, or click **Export Entire Library** to download all tools as a single JSON file.

### Importing
*   Click **Import Tool** and select a `.json` (NeuroCore export) or `.py` (Python script) file.
*   If importing a `.py` file, NeuroCore will attempt to wrap it as a tool, but you may need to adjust the parameter schema manually.