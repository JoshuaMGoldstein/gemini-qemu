#!/bin/bash

# Puppy Linux VM startup script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_DIR="$SCRIPT_DIR/.llmvm"
ISO_PATH="$VM_DIR/isos/BookwormPup64_10.0.11.iso"
DISK_PATH="$VM_DIR/vms/puppy.qcow2"
QMP_SOCK="$VM_DIR/qmp/puppy.sock"
PID_FILE="$VM_DIR/pids/puppy.pid"

# Remove old socket if exists
rm -f "$QMP_SOCK"

# Start VM with minimal resources
qemu-system-x86_64 \
    -enable-kvm \
    -machine q35,accel=kvm \
    -cpu host \
    -smp 2 \
    -m 2048M \
    -drive file="$DISK_PATH",if=virtio,format=qcow2 \
    -cdrom "$ISO_PATH" \
    -device virtio-tablet-pci \
    -vga vmware \
    -k en-us \
    -display vnc=127.0.0.1:1,share=ignore \
    -qmp unix:"$QMP_SOCK",server=on,wait=off \
    -qmp tcp:127.0.0.1:4444,server=on,wait=off \
    -netdev user,id=net0 \
    -device virtio-net,netdev=net0 \
    -boot order=cd \
    -daemonize \
    -pidfile "$PID_FILE"

echo "Puppy Linux VM started"
echo "VNC: 127.0.0.1:5901"
echo "QMP: 127.0.0.1:4444"
echo "PID: $(cat "$PID_FILE" 2>/dev/null || echo "unknown")"
