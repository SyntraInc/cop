# Tool Discovery Attack Framework

This framework allows you to test AI agents for tool discovery vulnerabilities. It attempts to discover tools and their capabilities through conversation with the target agent.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your tools in a YAML file (see `tool_config_example.yaml` for format).

## Running Tool Discovery Attacks

To run a tool discovery attack:

```bash
python main.py --mode tool_enumeration \
    --tool-config tool_config_example.yaml \
    --target-model gpt-4o \
    --attack-model grok \
    --helper-model grok \
    --judge-model gpt-4o
```

### Arguments

- `--mode tool_enumeration`: Runs in tool discovery mode
- `--tool-config`: Path to your tool configuration YAML file
- `--target-model`: Model to test (the agent being attacked)
- `--attack-model`: Model to use for generating attacks
- `--helper-model`: Model to use for generating follow-up questions
- `--judge-model`: Model to use for evaluating tool discovery

## Tool Configuration Format

Your tool configuration YAML should follow this format:

```yaml
agents:
  - name: "Agent Name"
    tools:
      - name: "tool_name"
        description: "Tool description"
        parameters:
          - name: "param_name"
            required: true/false
            description: "Parameter description"
```

## Output

The framework will:
1. Attempt to discover tools through conversation
2. Show discovery metrics (accuracy, completeness)
3. List discovered tools and their parameters
4. Attempt to use discovered tools



