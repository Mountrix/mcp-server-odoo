"""Simple HTTP wrapper for MCP server that bypasses Accept header validation issues.

This wrapper uses uvicorn to serve the FastMCP app directly but with a middleware
that normalizes Accept headers before they reach MCP validation.
"""

from starlette.types import Receive, Scope, Send
import uvicorn

from .server import OdooMCPServer
from .logging_config import get_logger

logger = get_logger(__name__)


class AcceptHeaderNormalizer:
    """ASGI middleware that normalizes Accept headers before MCP validation."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            # Normalize Accept header in scope
            headers = list(scope.get('headers', []))
            original_accept = None
            
            # Find and remove existing Accept header
            new_headers = []
            for k, v in headers:
                if k.lower() == b'accept':
                    original_accept = v
                    try:
                        accept_str = v.decode('utf-8') if isinstance(v, bytes) else str(v)
                        logger.info(f"🔧 [WRAPPER] Original Accept: '{accept_str}'")
                    except Exception:
                        logger.info(f"🔧 [WRAPPER] Original Accept: {v}")
                else:
                    new_headers.append((k, v))
            
            # Add normalized Accept header
            new_headers.append((b'accept', b'application/x-ndjson'))
            scope['headers'] = new_headers
            
            if original_accept is None:
                logger.info("🔧 [WRAPPER] Added Accept: application/x-ndjson")
            else:
                logger.info("🔧 [WRAPPER] Replaced Accept with: application/x-ndjson")
        
        # Continue with normalized headers
        await self.app(scope, receive, send)


async def run_http_wrapper(host: str = "0.0.0.0", port: int = 8001):
    """Run the HTTP wrapper server using uvicorn with header normalization."""
    # Initialize MCP server
    server = OdooMCPServer()
    
    # Establish connection
    server._ensure_connection()
    server._register_resources()
    server._register_tools()
    
    # CRITICAL: Patch StreamableHTTPSessionManager.handle_request BEFORE starting
    # This intercepts at the right level
    logger.info("🔧 [WRAPPER] Patching StreamableHTTPSessionManager.handle_request...")
    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        
        if hasattr(StreamableHTTPSessionManager, 'handle_request'):
            original_handle = StreamableHTTPSessionManager.handle_request
            if not hasattr(original_handle, '_accept_patched'):
                async def patched_handle_request(self_instance, scope, receive, send):
                    if scope.get('type') == 'http':
                        # Normalize Accept header
                        headers = list(scope.get('headers', []))
                        new_headers = [(k, v) for k, v in headers if k.lower() != b'accept']
                        new_headers.append((b'accept', b'application/x-ndjson'))
                        scope['headers'] = new_headers
                        logger.info("✅ [WRAPPER] Normalized Accept header in handle_request")
                    return await original_handle(self_instance, scope, receive, send)
                
                patched_handle_request._accept_patched = True
                StreamableHTTPSessionManager.handle_request = patched_handle_request
                logger.info("✅✅✅ [WRAPPER] Patched handle_request successfully")
            else:
                logger.info("✅ [WRAPPER] handle_request already patched")
    except Exception as e:
        logger.error(f"❌ [WRAPPER] Error patching handle_request: {e}", exc_info=True)
    
    logger.info(f"🚀 [WRAPPER] Starting HTTP wrapper server on {host}:{port}")
    logger.info("🚀 [WRAPPER] Using handle_request patching to normalize headers")
    logger.info("🚀 [WRAPPER] This bypasses streamable HTTP Accept header validation")
    
    # Use the original run_http method which will use run_streamable_http_async
    # but our patch will intercept at handle_request level
    await server.run_http(host=host, port=port)

