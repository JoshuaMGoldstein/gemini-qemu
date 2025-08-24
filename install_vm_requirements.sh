#!/bin/bash

# Comprehensive install script for Gemini CLI VNC Tools with TinyCore Linux VM
# Installs QEMU, downloads TinyCore Linux, and sets up automated VM

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VM_DIR="$PROJECT_DIR/.llmvm"
ISOS_DIR="$VM_DIR/isos"
VMS_DIR="$VM_DIR/vms"
PUPPY_VERSION="10.0.11"
PUPPY_ISO="BookwormPup64_10.0.11.iso"
PUPPY_URL="https://distro.ibiblio.org/puppylinux/puppy-bookwormpup/BookwormPup64/10.0.11/BookwormPup64_10.0.11.iso"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    else
        error "Cannot detect Linux distribution"
    fi
    log "Detected distribution: $DISTRO $VERSION"
}

# Install QEMU and dependencies based on distribution
install_qemu() {
    log "Installing QEMU and dependencies..."
    
    case $DISTRO in
        arch|endeavouros|manjaro)
            sudo pacman -S --needed --noconfirm qemu-full edk2-ovmf python python-pip python-virtualenv curl wget git gcc make cmake libgl mesa tk python-opencv tesseract tesseract-data-eng
            ;;
        ubuntu|debian|linuxmint)
            sudo apt update
            sudo apt install -y qemu-system qemu-utils ovmf python3 python3-pip python3-venv python3-dev curl wget git gcc g++ make cmake libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 python3-tk tesseract-ocr libtesseract-dev
            ;;
        fedora|centos|rhel)
            sudo dnf install -y qemu qemu-kvm edk2-ovmf python3 python3-pip python3-virtualenv python3-devel curl wget git gcc gcc-c++ make cmake mesa-libGL python3-tkinter tesseract tesseract-langpack-eng
            ;;
        opensuse*)
            sudo zypper install -y qemu qemu-ovmf-x86_64 python3 python3-pip python3-virtualenv python3-devel curl wget git gcc gcc-c++ make cmake Mesa-libGL1 python3-tk tesseract-ocr tesseract-ocr-traineddata-english
            ;;
        *)
            error "Unsupported distribution: $DISTRO. Please install QEMU manually."
            ;;
    esac
    
    # Verify QEMU installation
    if ! command -v qemu-system-x86_64 &> /dev/null; then
        error "QEMU installation failed or qemu-system-x86_64 not found"
    fi
    
    log "QEMU installed successfully"
}

# Install Python dependencies for VNC tools
install_python_deps() {
    log "Installing Python dependencies for VNC tools..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$SCRIPT_DIR/.vnc_env" ]; then
        python3 -m venv "$SCRIPT_DIR/.vnc_env"
    fi
    
    # Install dependencies
    "$SCRIPT_DIR/.vnc_env/bin/pip" install --upgrade pip setuptools wheel
    
    # Install VNC tools dependencies
    "$SCRIPT_DIR/.vnc_env/bin/pip" install vncdotool pillow requests python-dotenv numpy opencv-python opencv-python-headless replicate
    
    # Install PyTorch (CPU version by default, CUDA if available)
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
        "$SCRIPT_DIR/.vnc_env/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        log "Installing CPU-only PyTorch..."
        "$SCRIPT_DIR/.vnc_env/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install OmniParser requirements if available
    if [ -f "$SCRIPT_DIR/OmniParser/requirements.txt" ]; then
        log "Installing OmniParser dependencies..."
        "$SCRIPT_DIR/.vnc_env/bin/pip" install -r "$SCRIPT_DIR/OmniParser/requirements.txt"
    else
        log "Installing core ML dependencies..."
        "$SCRIPT_DIR/.vnc_env/bin/pip" install easyocr transformers ultralytics paddlepaddle paddleocr
    fi
    
    log "Python dependencies installed"
}

# Download OmniParser models if needed
download_omniparser_models() {
    log "Checking OmniParser models..."
    
    WEIGHTS_DIR="$SCRIPT_DIR/OmniParser/weights"
    
    if [ -d "$SCRIPT_DIR/OmniParser" ]; then
        if [ ! -d "$WEIGHTS_DIR" ]; then
            warn "OmniParser weights directory not found. Creating it..."
            mkdir -p "$WEIGHTS_DIR"
        fi
        
        # Check if models exist
        if [ -z "$(ls -A $WEIGHTS_DIR 2>/dev/null)" ]; then
            log "Downloading OmniParser models (this may take a while)..."
            
            # Download icon detect model
            if [ ! -f "$WEIGHTS_DIR/icon_detect.pt" ]; then
                log "Downloading icon detection model..."
                wget -q --show-progress -O "$WEIGHTS_DIR/icon_detect.pt" \
                    "https://huggingface.co/microsoft/OmniParser/resolve/main/icon_detect.pt" || \
                    warn "Failed to download icon detection model"
            fi
            
            # Download icon caption model  
            if [ ! -f "$WEIGHTS_DIR/icon_caption.pt" ]; then
                log "Downloading icon caption model..."
                wget -q --show-progress -O "$WEIGHTS_DIR/icon_caption.pt" \
                    "https://huggingface.co/microsoft/OmniParser/resolve/main/icon_caption.pt" || \
                    warn "Failed to download icon caption model"
            fi
            
            log "Model download complete"
        else
            log "OmniParser models already present"
        fi
    else
        warn "OmniParser directory not found - skipping model download"
    fi
}

# Create VM directories
setup_directories() {
    log "Setting up VM directories..."
    mkdir -p "$VM_DIR"/{isos,vms,pids,logs,qmp}
    log "VM directories created"
}

# Download Puppy Linux
download_puppy() {
    log "Downloading Puppy Linux BookwormPup64 ISO (~780MB)..."
    
    if [ -f "$ISOS_DIR/$PUPPY_ISO" ] && [ -s "$ISOS_DIR/$PUPPY_ISO" ]; then
        log "Puppy Linux already downloaded"
        return 0
    fi
    
    mkdir -p "$ISOS_DIR"
    
    # Download with progress bar
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$ISOS_DIR/$PUPPY_ISO" "$PUPPY_URL"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$ISOS_DIR/$PUPPY_ISO" "$PUPPY_URL"
    else
        error "Neither wget nor curl available for download"
    fi
    
    # Verify download
    if [ ! -f "$ISOS_DIR/$PUPPY_ISO" ] || [ ! -s "$ISOS_DIR/$PUPPY_ISO" ]; then
        error "Failed to download Puppy Linux"
    fi
    
    local size=$(du -h "$ISOS_DIR/$PUPPY_ISO" | cut -f1)
    log "Puppy Linux downloaded successfully ($size)"
}

# Create VM disk image
create_vm_disk() {
    log "Creating VM disk image..."
    
    local disk_path="$VMS_DIR/puppy.qcow2"
    
    if [ -f "$disk_path" ]; then
        warn "VM disk already exists, skipping creation"
        return 0
    fi
    
    mkdir -p "$VMS_DIR"
    
    # Create 2GB disk (plenty for Puppy Linux + save files)
    qemu-img create -f qcow2 "$disk_path" 2G
    
    log "VM disk created: $disk_path"
}

# Create Alpine autoinstall configuration  
create_autoinstall_config() {
    log "Creating Alpine autoinstall configuration..."
    
    local config_dir="$VM_DIR/autoinstall"
    mkdir -p "$config_dir"
    
    # Note: Alpine will boot to live installer by default
    # For testing QMP mouse input, the live system is sufficient
    log "Alpine will boot to live installer - no automated setup needed for testing"
}

# Create VM startup script
create_vm_script() {
    log "Creating VM startup script..."
    
    cat > "$PROJECT_DIR/start_vm.sh" << EOF
#!/bin/bash

# Puppy Linux VM startup script
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
VM_DIR="\$SCRIPT_DIR/.llmvm"
ISO_PATH="\$VM_DIR/isos/$PUPPY_ISO"
DISK_PATH="\$VM_DIR/vms/puppy.qcow2"
QMP_SOCK="\$VM_DIR/qmp/puppy.sock"
PID_FILE="\$VM_DIR/pids/puppy.pid"

# Remove old socket if exists
rm -f "\$QMP_SOCK"

# Start VM with minimal resources
qemu-system-x86_64 \\
    -enable-kvm \\
    -machine q35,accel=kvm \\
    -cpu host \\
    -smp 2 \\
    -m 4096M \\
    -drive file="\$DISK_PATH",if=virtio,format=qcow2 \\
    -cdrom "\$ISO_PATH" \\
    -device virtio-tablet-pci \\
    -vga vmware \\
    -k en-us \\
    -display vnc=127.0.0.1:1,share=ignore \\
    -qmp unix:"\$QMP_SOCK",server=on,wait=off \\
    -qmp tcp:127.0.0.1:4444,server=on,wait=off \\
    -netdev user,id=net0 \\
    -device virtio-net,netdev=net0 \\
    -boot order=cd \\
    -daemonize \\
    -pidfile "\$PID_FILE"

echo "Puppy Linux VM started"
echo "VNC: 127.0.0.1:5901"
echo "QMP: 127.0.0.1:4444"
echo "PID: \$(cat "\$PID_FILE" 2>/dev/null || echo "unknown")"
EOF

    chmod +x "$PROJECT_DIR/start_vm.sh"
    
    # Create stop script
    cat > "$PROJECT_DIR/stop_vm.sh" << EOF
#!/bin/bash

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="\$SCRIPT_DIR/.llmvm/pids/puppy.pid"

if [ -f "\$PID_FILE" ]; then
    PID=\$(cat "\$PID_FILE")
    if kill -0 "\$PID" 2>/dev/null; then
        echo "Stopping Puppy VM (PID: \$PID)..."
        kill "\$PID"
        rm -f "\$PID_FILE"
        echo "VM stopped"
    else
        echo "VM not running"
        rm -f "\$PID_FILE"
    fi
else
    echo "No PID file found"
fi
EOF

    chmod +x "$PROJECT_DIR/stop_vm.sh"
    
    log "VM scripts created: start_vm.sh, stop_vm.sh"
}

# Update VM configuration for Puppy Linux
update_vm_config() {
    log "Updating VM configuration..."
    
    cat > "$SCRIPT_DIR/vm_config.json" << EOF
{
  "vm_targets": {
    "local": {
      "host": "127.0.0.1",
      "vnc_port": 5901,
      "qmp_port": 4444,
      "description": "Puppy Linux VM"
    },
    "puppy": {
      "host": "127.0.0.1", 
      "vnc_port": 5901,
      "qmp_port": 4444,
      "description": "Puppy Linux VM"
    }
  }
}
EOF
    
    log "VM configuration updated"
}

# Copy Gemini settings
copy_gemini_settings() {
    log "Copying Gemini settings to bin folder..."
    
    local settings_source="$HOME/.gemini/settings.json"
    local settings_dest="$SCRIPT_DIR/settings.json"
    
    if [ -f "$settings_source" ]; then
        cp "$settings_source" "$settings_dest"
        log "Gemini settings copied from ~/.gemini/settings.json"
    else
        warn "Gemini settings not found at ~/.gemini/settings.json - creating default"
        cat > "$settings_dest" << EOF
{
  "apiKey": "",
  "model": "gemini-2.0-flash-001",
  "maxTokens": 8192,
  "temperature": 0.1
}
EOF
    fi
    
    # Create .env file template if it doesn't exist
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        log "Creating .env template for Replicate API fallback..."
        cat > "$SCRIPT_DIR/.env" << 'EOF'
# Replicate API Configuration (fallback only - local OmniParser used first)
# Only needed if local model fails
REPLICATE_API_TOKEN=your_replicate_api_token_here
EOF
        log "Optional: Add Replicate API token to .env file for fallback"
    fi
}

# Create usage instructions
create_instructions() {
    log "Creating usage instructions..."
    
    cat > "$PROJECT_DIR/README_VM.md" << EOF
# Puppy Linux VM Setup

## VM Details
- **OS**: Puppy Linux BookwormPup64 (~780MB)
- **RAM**: 256MB (minimal footprint)
- **Disk**: 2GB (persistent storage)
- **Network**: Enabled for package downloads

## Starting the VM
\`\`\`bash
./start_vm.sh
\`\`\`

## Stopping the VM
\`\`\`bash
./stop_vm.sh
\`\`\`

## Accessing the VM
- **VNC**: Connect to \`127.0.0.1:5901\` 
- **QMP**: TCP port 4444 for screenshots/control

## Installing Packages in TinyCore
Once the VM is running, open terminal and use:

\`\`\`bash
# Install Firefox browser
tce-load -wi firefox-ESR

# Install Python
tce-load -wi python3

# Install text editor
tce-load -wi text-editor

# Install development tools
tce-load -wi gcc
tce-load -wi nodejs
\`\`\`

## Testing Gemini CLI Tools
\`\`\`bash
# Take screenshot and analyze
echo '{"vm_target":"local"}' | ./bin/call_tool get_screenshot_description

# Send mouse clicks
echo '{"vm_target":"local", "clicks":[{"x":400, "y":300, "button":1, "action":"click"}]}' | ./bin/call_tool send_mouse_clicks

# Send keyboard input
echo '{"vm_target":"local", "actions":[{"action":"type", "text":"hello world"}]}' | ./bin/call_tool send_keyboard
\`\`\`

## Package Persistence
Packages installed with \`tce-load -wi\` will persist between reboots.
The VM automatically mounts persistent storage on startup.

## Troubleshooting
- If VM doesn't start, check QEMU installation: \`qemu-system-x86_64 --version\`
- If networking fails, TinyCore will still boot but package installation won't work
- VNC connections require a VNC viewer (try \`vncviewer 127.0.0.1:5901\`)
EOF
    
    log "Instructions created: README_VM.md"
}

# Main installation function
main() {
    log "Starting Gemini CLI VNC Tools installation..."
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        error "Please run this script as a regular user (it will use sudo when needed)"
    fi
    
    detect_distro
    install_qemu
    install_python_deps
    download_omniparser_models
    setup_directories
    download_puppy
    create_vm_disk
    create_vm_script
    update_vm_config
    copy_gemini_settings
    create_instructions
    
    log "Installation completed successfully!"
    echo
    echo -e "${GREEN}Starting Puppy Linux VM...${NC}"
    cd "$PROJECT_DIR"
    ./start_vm.sh
    echo
    echo -e "${GREEN}Next steps:${NC}"
    echo -e "1. Connect via VNC: ${BLUE}vncviewer 127.0.0.1:5901${NC}"
    echo -e "2. Test tools: ${BLUE}echo '{\"vm_target\":\"local\"}' | ./bin/call_tool get_screenshot_description${NC}"
    echo "3. Read README_VM.md for detailed usage instructions"
    echo
    echo -e "${YELLOW}Note: Puppy Linux will boot to full GUI desktop. Perfect for testing QMP mouse input!${NC}"
}

# Run main function
main "$@"