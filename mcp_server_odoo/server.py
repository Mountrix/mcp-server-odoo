"""MCP Server implementation for Odoo.

This module provides the FastMCP server that exposes Odoo data
and functionality through the Model Context Protocol.
"""

from typing import Any, Dict, Optional

from mcp.server import FastMCP

from .access_control import AccessController
from .config import OdooConfig, get_config
from .error_handling import (
    ConfigurationError,
    ErrorContext,
    error_handler,
)
from .logging_config import get_logger, logging_config, perf_logger
from .odoo_connection import OdooConnection, OdooConnectionError
from .performance import PerformanceManager
from .resources import register_resources
from .tools import register_tools

# Set up logging
logger = get_logger(__name__)

# Server version
SERVER_VERSION = "0.1.0"


class OdooMCPServer:
    """Main MCP server class for Odoo integration.

    This class manages the FastMCP server instance and maintains
    the connection to Odoo. The server lifecycle is managed by
    establishing connection before starting and cleaning up on exit.
    """

    def __init__(self, config: Optional[OdooConfig] = None):
        """Initialize the Odoo MCP server.

        Args:
            config: Optional OdooConfig instance. If not provided,
                   will load from environment variables.
        """
        # Load configuration
        self.config = config or get_config()

        # Set up structured logging
        logging_config.setup()

        # Initialize connection and access controller (will be created on startup)
        self.connection: Optional[OdooConnection] = None
        self.access_controller: Optional[AccessController] = None
        self.performance_manager: Optional[PerformanceManager] = None
        self.resource_handler = None
        self.tool_handler = None

        # Create FastMCP instance with server metadata
        self.app = FastMCP(
            name="odoo-mcp-server",
            instructions="MCP server for accessing and managing Odoo ERP data through the Model Context Protocol",
        )

        logger.info(f"Initialized Odoo MCP Server v{SERVER_VERSION}")

    def _ensure_connection(self):
        """Ensure connection to Odoo is established.

        Raises:
            ConnectionError: If connection fails
            ConfigurationError: If configuration is invalid
        """
        if not self.connection:
            try:
                logger.info("Establishing connection to Odoo...")
                with perf_logger.track_operation("connection_setup"):
                    # Create performance manager (shared across components)
                    self.performance_manager = PerformanceManager(self.config)

                    # Create connection with performance manager
                    self.connection = OdooConnection(
                        self.config, performance_manager=self.performance_manager
                    )

                    # Connect and authenticate
                    self.connection.connect()
                    self.connection.authenticate()

                logger.info(f"Successfully connected to Odoo at {self.config.url}")

                # Initialize access controller
                self.access_controller = AccessController(self.config)
            except Exception as e:
                context = ErrorContext(operation="connection_setup")
                # Let specific errors propagate as-is
                if isinstance(e, (OdooConnectionError, ConfigurationError)):
                    raise
                # Handle other unexpected errors
                error_handler.handle_error(e, context=context)

    def _cleanup_connection(self):
        """Clean up Odoo connection."""
        if self.connection:
            try:
                logger.info("Closing Odoo connection...")
                self.connection.disconnect()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                # Always clear connection reference
                self.connection = None
                self.access_controller = None
                self.resource_handler = None
                self.tool_handler = None

    def _setup_handlers(self):
        """Set up MCP handlers for resources, tools, and prompts.

        This method will be extended in later phases to add:
        - Resource handlers for Odoo data access
        - Tool handlers for Odoo operations
        - Prompt handlers for guided workflows
        """
        # TODO: Tools will be added in Phase 3
        # TODO: Prompts will be added in Phase 4
        pass

    def _register_resources(self):
        """Register resource handlers after connection is established."""
        if self.connection and self.access_controller:
            self.resource_handler = register_resources(
                self.app, self.connection, self.access_controller, self.config
            )
            logger.info("Registered MCP resources")

    def _register_tools(self):
        """Register tool handlers after connection is established."""
        if self.connection and self.access_controller:
            self.tool_handler = register_tools(
                self.app, self.connection, self.access_controller, self.config
            )
            logger.info("Registered MCP tools")

    async def run_stdio(self):
        """Run the server using stdio transport.

        This is the main entry point for running the server
        with standard input/output transport (used by uvx).
        """
        try:
            # Establish connection before starting server
            with perf_logger.track_operation("server_startup"):
                self._ensure_connection()

                # Register resources after connection is established
                self._register_resources()
                self._register_tools()

            logger.info("Starting MCP server with stdio transport...")
            await self.app.run_stdio_async()

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except (OdooConnectionError, ConfigurationError):
            # Let these specific errors propagate
            raise
        except Exception as e:
            context = ErrorContext(operation="server_run")
            error_handler.handle_error(e, context=context)
        finally:
            # Always cleanup connection
            self._cleanup_connection()

    def run_stdio_sync(self):
        """Synchronous wrapper for run_stdio.

        This is provided for compatibility with synchronous code.
        """
        import asyncio

        asyncio.run(self.run_stdio())

    # SSE transport has been deprecated in MCP protocol version 2025-03-26
    # Use streamable-http transport instead

    async def run_http(self, host: str = "localhost", port: int = 8000):
        """Run the server using streamable HTTP transport.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        try:
            # Establish connection before starting server
            with perf_logger.track_operation("server_startup"):
                self._ensure_connection()

                # Register resources after connection is established
                self._register_resources()
                self._register_tools()

            logger.info(f"Starting MCP server with HTTP transport on {host}:{port}...")
            
            # CRITICAL FIX: Add middleware to force Accept header before MCP validation
            logger.info("🔧 [ACCEPT HEADER FIX] Adding middleware to force Accept header...")
            try:
                from starlette.middleware.base import BaseHTTPMiddleware
                from starlette.requests import Request
                from starlette.responses import Response
                
                class AcceptHeaderFixMiddleware(BaseHTTPMiddleware):
                    """Middleware to force Accept header to application/x-ndjson for MCP compatibility."""
                    
                    async def dispatch(self, request: Request, call_next):
                        # Log original Accept header
                        original_accept = request.headers.get('accept', '<not found>')
                        path = request.url.path
                        method = request.method
                        logger.info(f"🔍 [ACCEPT FIX] Request: {method} {path}")
                        logger.info(f"🔍 [ACCEPT FIX] Original Accept header: {original_accept}")
                        logger.info(f"🔍 [ACCEPT FIX] All headers: {dict(request.headers)}")
                        
                        # Force Accept header to application/x-ndjson
                        if 'accept' not in request.headers:
                            logger.info("🔧 [ACCEPT FIX] No Accept header found, adding application/x-ndjson")
                            request.headers['accept'] = 'application/x-ndjson'
                        elif 'application/x-ndjson' not in request.headers.get('accept', '').lower():
                            logger.info(f"🔧 [ACCEPT FIX] Accept header doesn't contain application/x-ndjson, replacing")
                            logger.info(f"🔧 [ACCEPT FIX] Old Accept: {request.headers.get('accept')}")
                            request.headers['accept'] = 'application/x-ndjson'
                            logger.info(f"🔧 [ACCEPT FIX] New Accept: {request.headers.get('accept')}")
                        else:
                            logger.info("✅ [ACCEPT FIX] Accept header already contains application/x-ndjson")
                        
                        # Verify the change
                        final_accept = request.headers.get('accept', '<not found>')
                        logger.info(f"✅ [ACCEPT FIX] Final Accept header before processing: {final_accept}")
                        
                        # Process request
                        try:
                            response = await call_next(request)
                            logger.info(f"✅ [ACCEPT FIX] Request processed successfully, status: {response.status_code}")
                            return response
                        except Exception as e:
                            logger.error(f"❌ [ACCEPT FIX] Error processing request: {e}", exc_info=True)
                            raise
                
                # Add middleware to the FastMCP app
                logger.info("🔧 [ACCEPT FIX] Attempting to add middleware to FastMCP app...")
                logger.info(f"🔧 [ACCEPT FIX] App type: {type(self.app)}")
                logger.info(f"🔧 [ACCEPT FIX] App attributes: {[a for a in dir(self.app) if not a.startswith('_')][:20]}")
                
                # Try to get the underlying Starlette/FastAPI app
                if hasattr(self.app, 'app'):
                    underlying_app = self.app.app
                    logger.info(f"🔧 [ACCEPT FIX] Found underlying app: {type(underlying_app)}")
                    
                    # Add middleware
                    if hasattr(underlying_app, 'add_middleware'):
                        underlying_app.add_middleware(AcceptHeaderFixMiddleware)
                        logger.info("✅✅✅ [ACCEPT FIX] SUCCESS: Added middleware via add_middleware")
                    elif hasattr(underlying_app, 'middleware'):
                        underlying_app.middleware('http')(AcceptHeaderFixMiddleware)
                        logger.info("✅✅✅ [ACCEPT FIX] SUCCESS: Added middleware via @middleware decorator")
                    else:
                        logger.warning("⚠️ [ACCEPT FIX] App doesn't support add_middleware or @middleware, trying direct patch")
                        # Direct patch: wrap the app's __call__ method
                        if hasattr(underlying_app, '__call__'):
                            original_call = underlying_app.__call__
                            async def patched_call(scope, receive, send):
                                if scope.get('type') == 'http':
                                    path = scope.get('path', 'unknown')
                                    method = scope.get('method', 'unknown')
                                    logger.info(f"🔍 [ACCEPT FIX ASGI] Intercepted: {method} {path}")
                                    
                                    # Get original headers
                                    original_headers = dict(scope.get('headers', []))
                                    original_accept = original_headers.get(b'accept', b'<not found>')
                                    logger.info(f"🔍 [ACCEPT FIX ASGI] Original Accept: {original_accept}")
                                    
                                    # Force Accept header
                                    headers = []
                                    accept_found = False
                                    for k, v in scope.get('headers', []):
                                        if k.lower() == b'accept':
                                            headers.append((b'accept', b'application/x-ndjson'))
                                            accept_found = True
                                            logger.info(f"🔧 [ACCEPT FIX ASGI] Replaced Accept: {v} -> application/x-ndjson")
                                        else:
                                            headers.append((k, v))
                                    
                                    if not accept_found:
                                        headers.append((b'accept', b'application/x-ndjson'))
                                        logger.info("🔧 [ACCEPT FIX ASGI] Added Accept: application/x-ndjson")
                                    
                                    scope['headers'] = headers
                                    logger.info(f"✅ [ACCEPT FIX ASGI] Modified scope headers count: {len(headers)}")
                                
                                return await original_call(scope, receive, send)
                            
                            underlying_app.__call__ = patched_call
                            logger.info("✅✅✅ [ACCEPT FIX] SUCCESS: Patched app.__call__ directly")
                else:
                    logger.warning("⚠️ [ACCEPT FIX] App doesn't have 'app' attribute, trying to patch self.app directly")
                    if hasattr(self.app, '__call__'):
                        original_call = self.app.__call__
                        async def patched_app_call(scope, receive, send):
                            if scope.get('type') == 'http':
                                path = scope.get('path', 'unknown')
                                method = scope.get('method', 'unknown')
                                logger.info(f"🔍 [ACCEPT FIX APP] Intercepted: {method} {path}")
                                
                                headers = []
                                accept_found = False
                                for k, v in scope.get('headers', []):
                                    if k.lower() == b'accept':
                                        headers.append((b'accept', b'application/x-ndjson'))
                                        accept_found = True
                                        logger.info(f"🔧 [ACCEPT FIX APP] Replaced Accept: {v} -> application/x-ndjson")
                                    else:
                                        headers.append((k, v))
                                
                                if not accept_found:
                                    headers.append((b'accept', b'application/x-ndjson'))
                                    logger.info("🔧 [ACCEPT FIX APP] Added Accept: application/x-ndjson")
                                
                                scope['headers'] = headers
                                logger.info(f"✅ [ACCEPT FIX APP] Modified scope headers")
                            
                            return await original_call(scope, receive, send)
                        
                        self.app.__call__ = patched_app_call
                        logger.info("✅✅✅ [ACCEPT FIX] SUCCESS: Patched self.app.__call__ directly")
                
            except Exception as e:
                logger.error(f"❌ [ACCEPT FIX] ERROR adding middleware: {e}", exc_info=True)
                logger.error("⚠️ [ACCEPT FIX] Continuing without middleware fix (may cause 406 errors)")

            # Update FastMCP settings for host and port
            self.app.settings.host = host
            self.app.settings.port = port

            logger.info("🚀 [ACCEPT FIX] Starting server with Accept header fix applied...")
            # Use the specific streamable HTTP async method
            await self.app.run_streamable_http_async()

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except (OdooConnectionError, ConfigurationError):
            # Let these specific errors propagate
            raise
        except Exception as e:
            context = ErrorContext(operation="server_run_http")
            error_handler.handle_error(e, context=context)
        finally:
            # Always cleanup connection
            self._cleanup_connection()

    def get_capabilities(self) -> Dict[str, Dict[str, bool]]:
        """Get server capabilities.

        Returns:
            Dict with server capabilities
        """
        return {
            "capabilities": {
                "resources": True,  # Exposes Odoo data as resources
                "tools": True,  # Provides tools for Odoo operations
                "prompts": False,  # Prompts will be added in later phases
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get server health status with error metrics.

        Returns:
            Dict with health status and metrics
        """
        is_connected = (
            self.connection and self.connection.is_authenticated
            if hasattr(self.connection, "is_authenticated")
            else False
        )

        # Get performance stats if available
        performance_stats = None
        if self.performance_manager:
            performance_stats = self.performance_manager.get_stats()

        return {
            "status": "healthy" if is_connected else "unhealthy",
            "version": SERVER_VERSION,
            "connection": {
                "connected": is_connected,
                "url": self.config.url if self.config else None,
                "database": (
                    self.connection.database
                    if self.connection and hasattr(self.connection, "database")
                    else None
                ),
            },
            "error_metrics": error_handler.get_metrics(),
            "recent_errors": error_handler.get_recent_errors(limit=5),
            "performance": performance_stats,
        }
