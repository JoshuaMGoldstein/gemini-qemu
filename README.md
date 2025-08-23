gemini-qemu, install everything and start the vm with install_vm_requirements.sh
first screenshot tool use will take a couple minutes to install everything.  

How to use:  After modifying your .gemini settings.json to specify the tools location, as
in the settings.json example, tell gemini to do something with the vm .  It should properly
use the screenshot tool, and the mouse click and keyboard input tools.  Prompting will be required
for it to be intelligent about it - perhaps modify system prompt to gemini to tell it to 
always take a screenshot after doing an action to verify it worked

API KEYS:  will use local model for screenshot detection, but there's a fallback for replicate that
can be used by exporting environment variable REPLICATE_API_TOKEN

