# Puppy Linux VM Setup

## VM Details
- **OS**: Puppy Linux BookwormPup64 (~780MB)
- **RAM**: 256MB (minimal footprint)
- **Disk**: 2GB (persistent storage)
- **Network**: Enabled for package downloads

## Starting the VM
```bash
./start_vm.sh
```

## Stopping the VM
```bash
./stop_vm.sh
```

## Accessing the VM
- **VNC**: Connect to `127.0.0.1:5901` 
- **QMP**: TCP port 4444 for screenshots/control

## Installing Packages in TinyCore
Once the VM is running, open terminal and use:

```bash
# Install Firefox browser
tce-load -wi firefox-ESR

# Install Python
tce-load -wi python3

# Install text editor
tce-load -wi text-editor

# Install development tools
tce-load -wi gcc
tce-load -wi nodejs
```

## Testing Gemini CLI Tools
```bash
# Take screenshot and analyze
echo '{"vm_target":"local"}' | ./bin/call_tool get_screenshot_description

# Send mouse clicks
echo '{"vm_target":"local", "clicks":[{"x":400, "y":300, "button":1, "action":"click"}]}' | ./bin/call_tool send_mouse_clicks

# Send keyboard input
echo '{"vm_target":"local", "actions":[{"action":"type", "text":"hello world"}]}' | ./bin/call_tool send_keyboard
```

## Package Persistence
Packages installed with `tce-load -wi` will persist between reboots.
The VM automatically mounts persistent storage on startup.

## Troubleshooting
- If VM doesn't start, check QEMU installation: `qemu-system-x86_64 --version`
- If networking fails, TinyCore will still boot but package installation won't work
- VNC connections require a VNC viewer (try `vncviewer 127.0.0.1:5901`)
