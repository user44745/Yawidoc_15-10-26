"""
Open-LLM-VTuber Server
========================
This module contains the WebSocket server for Open-LLM-VTuber, which handles
the WebSocket connections, serves static files, and manages the web tool.
It uses FastAPI for the server and Starlette for static file serving.
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, Response
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

from .routes import (
    init_client_ws_route,
    init_webtool_routes,
    init_proxy_route,
    init_interpreter_route,
)
from .service_context import ServiceContext
from .config_manager.utils import Config


# Create a custom StaticFiles class that adds CORS headers
class CORSStaticFiles(StarletteStaticFiles):
    """
    Static files handler that adds CORS headers to all responses.
    Needed because Starlette StaticFiles might bypass standard middleware.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        # Add CORS headers to all responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

        if path.endswith(".js"):
            response.headers["Content-Type"] = "application/javascript"

        return response


class AvatarStaticFiles(CORSStaticFiles):
    """
    Avatar files handler with security restrictions and CORS headers
    """

    async def get_response(self, path: str, scope):
        allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".svg")
        if not any(path.lower().endswith(ext) for ext in allowed_extensions):
            return Response("Forbidden file type", status_code=403)
        response = await super().get_response(path, scope)
        return response


class FrontendStaticFiles(CORSStaticFiles):
    """
    The frontend, with the quick-phrase bar on its page. The bar is a script
    of its own (frontend_addons/quick_bar.js) that frontend/index.html loads.
    The frontend is a submodule: when an update of it brings back an
    index.html without the script, the script is added as the page is served.
    The page is never served from the browser's cache, so that a change to
    it shows on the next load.
    """

    ADDONS = '<script src="/frontend-addons/quick_bar.js"></script>'

    async def get_response(self, path: str, scope):
        if path in (".", "index.html"):
            page = Path(self.directory) / "index.html"
            if page.is_file():
                html = page.read_text(encoding="utf-8")
                if "quick_bar.js" not in html:
                    # before the app's own script, so that it sees the connection open
                    html = html.replace("<head>", f"<head>\n    {self.ADDONS}", 1)
                return HTMLResponse(
                    html,
                    headers={
                        "Cache-Control": "no-cache",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, OPTIONS",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
        return await super().get_response(path, scope)


class WebSocketServer:
    """
    API server for Open-LLM-VTuber. This contains the websocket endpoint for the client, hosts the web tool, and serves static files.

    Creates and configures a FastAPI app, registers all routes
    (WebSocket, web tools, proxy) and mounts static assets with CORS.

    Args:
        config (Config): Application configuration containing system settings.
        default_context_cache (ServiceContext, optional):
            Pre‑initialized service context for sessions' service context to reference to.
            **If omitted, `initialize()` method needs to be called to load service context.**

    Notes:
        - If default_context_cache is omitted, call `await initialize()` to load service context cache.
        - Use `clean_cache()` to clear and recreate the local cache directory.
    """

    def __init__(self, config: Config, default_context_cache: ServiceContext = None):
        self.app = FastAPI(title="Open-LLM-VTuber Server")  # Added title for clarity
        self.config = config
        self.default_context_cache = (
            default_context_cache or ServiceContext()
        )  # Use provided context or initialize a new empty one waiting to be loaded
        # It will be populated during the initialize method call

        # Add global CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Include routes, passing the context instance
        # The context will be populated during the initialize step
        self.app.include_router(
            init_client_ws_route(default_context_cache=self.default_context_cache),
        )
        self.app.include_router(
            init_webtool_routes(default_context_cache=self.default_context_cache),
        )
        self.app.include_router(init_interpreter_route())

        # Initialize and include proxy routes if proxy is enabled
        system_config = config.system_config
        if hasattr(system_config, "enable_proxy") and system_config.enable_proxy:
            # Construct the server URL for the proxy
            host = system_config.host
            port = system_config.port
            server_url = f"ws://{host}:{port}/client-ws"
            self.app.include_router(
                init_proxy_route(server_url=server_url),
            )

        # Mount cache directory first (to ensure audio file access)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        self.app.mount(
            "/cache",
            CORSStaticFiles(directory="cache"),
            name="cache",
        )

        # Mount static files with CORS-enabled handlers
        self.app.mount(
            "/live2d-models",
            CORSStaticFiles(directory="live2d-models"),
            name="live2d-models",
        )
        self.app.mount(
            "/bg",
            CORSStaticFiles(directory="backgrounds"),
            name="backgrounds",
        )
        self.app.mount(
            "/avatars",
            AvatarStaticFiles(directory="avatars"),
            name="avatars",
        )

        # Mount web tool directory separately from frontend
        self.app.mount(
            "/web-tool",
            CORSStaticFiles(directory="web_tool", html=True),
            name="web_tool",
        )

        # Mount the interpreter page (doctor / patient)
        self.app.mount(
            "/interpreter",
            CORSStaticFiles(directory="interpreter", html=True),
            name="interpreter",
        )

        # Mount the additions to the frontend page (quick-phrase bar)
        self.app.mount(
            "/frontend-addons",
            CORSStaticFiles(directory="frontend_addons"),
            name="frontend_addons",
        )

        # Mount main frontend last (as catch-all)
        self.app.mount(
            "/",
            FrontendStaticFiles(directory="frontend", html=True),
            name="frontend",
        )

    async def initialize(self):
        """Asynchronously load the service context from config.
        Calling this function is needed if default_context_cache was not provided to the constructor."""
        await self.default_context_cache.load_from_config(self.config)

    @staticmethod
    def clean_cache():
        """Clean the cache directory by removing and recreating it."""
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
