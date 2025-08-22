#!/usr/bin/env python3
"""
Simple QMP (QEMU Machine Protocol) client for network screenshots
"""

import json
import socket
import tempfile
from typing import Dict, Any
from pathlib import Path


class QMPClient:
    """Simple QMP client for network connections"""
    
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        
    def connect(self):
        """Connect to QMP server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        self.socket.connect((self.host, self.port))
        
        # Read QMP greeting
        greeting = self._read_message()
        if 'QMP' not in greeting:
            raise Exception(f"Invalid QMP greeting: {greeting}")
            
        # Send qmp_capabilities command
        self._send_command('qmp_capabilities')
        
    def disconnect(self):
        """Disconnect from QMP server"""
        if self.socket:
            try:
                self.socket.close()
            finally:
                self.socket = None
                
    def screendump(self, filename: str) -> bool:
        """Take a screenshot via QMP screendump command"""
        try:
            response = self._send_command('screendump', {'filename': filename})
            return response.get('return') == {}
        except Exception:
            return False
            
    def send_mouse_event(self, x: int, y: int, button: int, pressed: bool) -> bool:
        """Send mouse event via QMP"""
        try:
            # Convert screen coordinates to QMP coordinates (32767 max range)
            # Use actual VMware VGA resolution for conversion
            width, height = 800, 600
            qmp_x = int((x / width) * 32767)
            qmp_y = int((y / height) * 32767)
            
            # Clamp to valid range
            qmp_x = max(0, min(qmp_x, 32767))
            qmp_y = max(0, min(qmp_y, 32767))
            
            # Combine movement and button events in single command
            events = [
                {'type': 'abs', 'data': {'axis': 'x', 'value': qmp_x}},
                {'type': 'abs', 'data': {'axis': 'y', 'value': qmp_y}}
            ]
            
            # Add button event if specified
            if button > 0:
                button_map = {1: 'left', 2: 'middle', 3: 'right'}
                btn_name = button_map.get(button, 'left')
                events.append({'type': 'btn', 'data': {'button': btn_name, 'down': pressed}})
            
            response = self._send_command('input-send-event', {'events': events})
            return response.get('return') == {}
            
        except Exception:
            return False
            
    def send_key_event(self, key: str, pressed: bool) -> bool:
        """Send keyboard event via QMP"""
        try:
            events = [
                {
                    'type': 'key',
                    'data': {
                        'key': {'type': 'qcode', 'data': key},
                        'down': pressed
                    }
                }
            ]
            response = self._send_command('input-send-event', {'events': events})
            return response.get('return') == {}
        except Exception:
            return False
            
    def _send_command(self, command: str, arguments: Dict = None) -> Dict[str, Any]:
        """Send a QMP command and return response"""
        cmd = {'execute': command}
        if arguments:
            cmd['arguments'] = arguments
            
        # Send command
        cmd_json = json.dumps(cmd) + '\n'
        self.socket.send(cmd_json.encode('utf-8'))
        
        # Read response
        response = self._read_message()
        
        if 'error' in response:
            raise Exception(f"QMP error: {response['error']}")
            
        return response
        
    def _read_message(self) -> Dict[str, Any]:
        """Read a JSON message from QMP socket"""
        buffer = b''
        while b'\n' not in buffer:
            data = self.socket.recv(1024)
            if not data:
                raise Exception("QMP connection closed")
            buffer += data
            
        # Parse the first complete JSON message
        line = buffer.split(b'\n')[0]
        return json.loads(line.decode('utf-8'))


def qmp_screenshot(host: str, port: int, output_path: str) -> bool:
    """Take a screenshot using QMP over network"""
    try:
        client = QMPClient(host, port)
        client.connect()
        
        success = client.screendump(output_path)
        client.disconnect()
        
        # Verify file was created and has content
        if success and Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            return file_size > 1024  # Must be > 1KB like the working code
            
        return False
        
    except Exception as e:
        return False


if __name__ == "__main__":
    # Test QMP screenshot
    import sys
    if len(sys.argv) == 4:
        host, port, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
        success = qmp_screenshot(host, port, output)
        print(f"Screenshot {'successful' if success else 'failed'}")
    else:
        print("Usage: qmp_client.py <host> <port> <output_file>")