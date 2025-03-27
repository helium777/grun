# grun - GPU Resource Manager

A simple command-line tool to run scripts when GPU resources are available.

## Features

- Monitor GPU memory and utilization using NVIDIA Management Library (NVML)
- Run commands when sufficient GPU resources are available
- Smart GPU selection strategies:
  - Default: Select GPUs with lower utilization
  - Exclusive: Only use GPUs with no other running processes
- Detailed GPU status reporting (memory, utilization, active processes)
- Configurable check interval
- Option to occupy GPU memory and keep it busy with computation
- Notification support for task status (using Bark or Telegram)

## Requirements

- Python 3.9+
- NVIDIA GPU with NVIDIA drivers installed
- CUDA toolkit (for GPU occupation feature)

## Installation

Since this package is currently in demo stage and not available on PyPI, you can install it directly from GitHub using `uv`:

```bash
# Install using uv
uv tool install git+https://github.com/helium777/grun.git

# If you want to use the GPU occupation feature, install with the 'occupy' extra
uv tool install "grun[occupy] @ git+https://github.com/helium777/grun.git"
```

Alternatively, you can install it using pip:

```bash
# Install using pip
pip install git+https://github.com/helium7/grun.git

# If you want to use the GPU occupation feature, install with the 'occupy' extra
pip install "grun[occupy] @ git+https://github.com/helium7/grun.git"
```

## Usage

Basic syntax:
```bash
grun --mem <memory_in_gb> [--strategy <strategy>] <command>
```

Available strategies:
- `utilization` (default): Prefer GPUs with lower utilization
- `exclusive`: Only use GPUs with no other processes

### Examples

1. Run a training script using default strategy (lowest utilization):
```bash
grun --mem 40 python train.py
```

2. Run a script on exclusive GPUs:
```bash
grun --mem 32 --strategy exclusive python train.py
```

3. Run a script with custom check interval (e.g., check every 5 seconds):
```bash
grun --mem 32 --interval 5 python my_script.py
```

4. Run a command with arguments:
```bash
grun --mem 16 python train.py --batch-size 32 --epochs 100
```

5. Occupy GPU memory and keep it busy:
```bash
grun --mem 16 --occupy
```

## Configuration

The tool uses a configuration file located at `~/.config/grun/config.toml` to store settings. You can create or modify this file to customize the behavior.

### Notification Settings

You can configure notifications using the following settings in the config file:

```toml
[notification]
service = "bark"  # Options: "bark", "telegram", "slack", or "none" to disable notifications
notify_on_gpu_found = true  # Whether to notify when GPUs are found
notify_on_task_complete = true  # Whether to notify when task completes

# Bark notification settings (if service = "bark")
[notification.bark]
key = "your-bark-key"  # Your Bark app key
server = "https://api.day.app"  # Optional: custom Bark server

# Telegram notification settings (if service = "telegram")
[notification.telegram]
bot_token = "your-bot-token"  # Your Telegram bot token
chat_id = "your-chat-id"  # Your Telegram chat ID

# Slack notification settings (if service = "slack")
[notification.slack]
webhook_url = "your-webhook-url"  # Your Slack webhook URL
```

#### Setting up Bark notifications:
1. Install the Bark app on your iOS device
2. Get your Bark key from the app
3. Add it to the config file as shown above

#### Setting up Telegram notifications:
1. Create a new Telegram bot using [@BotFather](https://t.me/botfather) and get the bot token
2. Start a chat with your bot and get your chat ID using [@userinfobot](https://t.me/userinfobot)
3. Add the bot token and chat ID to the config file

#### Setting up Slack notifications:
1. Create a new Slack app in your workspace at https://api.slack.com/apps
2. Enable "Incoming Webhooks" in your app's features
3. Create a new webhook for your desired channel
4. Copy the webhook URL and add it to the config file

## How it works

1. The tool uses NVIDIA Management Library (NVML) to monitor:
   - Available GPU memory
   - GPU utilization
   - Running processes
2. Based on the selected strategy:
   - Default (`utilization`): Selects GPUs with lowest utilization and sufficient memory
   - `exclusive`: Only selects GPUs with no other processes running
3. When suitable GPUs are found:
   - Sets the `CUDA_VISIBLE_DEVICES` environment variable
   - Either runs the specified command or occupies the GPU (if --occupy is used)
4. The process continues until the command completes or is interrupted

## Notes

- Memory requirement is specified in gigabytes (GB)
- The tool will wait indefinitely until suitable GPUs are found
- Use Ctrl+C to stop waiting for resources or to stop GPU occupation
- The tool properly initializes and cleans up NVML resources
- When using --occupy, the tool will:
  - Allocate the specified amount of GPU memory
  - Keep the GPU busy with continuous computation
  - Clean up resources when interrupted with Ctrl+C

## Development

To contribute to the development of `grun`:

1. Clone the repository
2. Install development dependencies:
```bash
pip install -e ".[occupy]"
```
