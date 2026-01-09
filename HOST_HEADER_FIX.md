# Fix for Host Header Validation Issue

## Problem

The MCP server returns `421 Misdirected Request` because the `mcp` library (FastMCP) validates the Host header strictly. When running behind nginx in Kubernetes, the server sees the pod IP instead of the service name.

## Root Cause

The validation happens in the `mcp` library's `transport_security` module, which we don't control. The library validates the Host header against the connection's IP address, not just the Host header value.

## Solution

You need to modify the `mcp` library source code to disable Host header validation when behind a proxy.

### Option 1: Fork and Modify mcp Library

1. Fork the `mcp` library repository
2. Find the `transport_security` module that validates Host headers
3. Add a configuration option to disable Host validation when `MCP_DISABLE_HOST_VALIDATION=true`
4. Install your forked version in the Docker image

### Option 2: Patch the Installed Library

You can patch the installed library in the Dockerfile:

```dockerfile
# After installing mcp-server-odoo
RUN python -c "
import re
import pathlib

# Find the transport_security module
mcp_path = pathlib.Path('/usr/local/lib/python3.12/site-packages/mcp')
transport_security_file = mcp_path / 'server' / 'transport_security.py'

if transport_security_file.exists():
    content = transport_security_file.read_text()
    # Disable Host validation by modifying the validation logic
    # This is a hack and may break with library updates
    modified = re.sub(
        r'if.*host.*validation',
        r'if False and host_validation',  # Disable validation
        content
    )
    transport_security_file.write_text(modified)
"
```

### Option 3: Use a Different Approach

Instead of trying to disable Host validation, ensure nginx sends the Host header in a format that the server accepts. However, this may not work if the server validates against the connection IP.

## Current Status

The code in `server.py` has been simplified to remove the non-working uvicorn direct approach. The `MCP_DISABLE_HOST_VALIDATION` environment variable is set in the Kubernetes deployment but won't have any effect until the `mcp` library is modified to support it.

## Recommended Next Steps

1. Contact the `mcp` library maintainers to request a feature to disable Host validation
2. Or fork the library and add the feature yourself
3. Or use a service mesh (Istio/Linkerd) that handles Host headers better

