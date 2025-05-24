from typing import Any, Dict, Optional, AsyncIterator, Callable, TypeVar, Awaitable
import os
import httpx
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps

from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)

# Constants
SUPERSET_BASE_URL = os.getenv("SUPERSET_BASE_URL", "http://localhost:8088")
SUPERSET_USERNAME = os.getenv("SUPERSET_USERNAME")
SUPERSET_PASSWORD = os.getenv("SUPERSET_PASSWORD")

# Define application lifespan context
@dataclass
class SupersetContext:
    client: httpx.AsyncClient
    base_url: str
    access_token: Optional[str] = None
    csrf_token: Optional[str] = None

@asynccontextmanager
async def superset_lifespan(server: FastMCP) -> AsyncIterator[SupersetContext]:
    client = httpx.AsyncClient(base_url=SUPERSET_BASE_URL, timeout=30.0)
    ctx = SupersetContext(client=client, base_url=SUPERSET_BASE_URL)
    try:
        yield ctx
    finally:
        await client.aclose()

# Initialize MCP server with the Superset lifespan
mcp = FastMCP(
    "superset",
    lifespan=superset_lifespan,
    dependencies=["fastapi", "uvicorn", "python-dotenv", "httpx"],
)

# Type variables for generic annotations
T = TypeVar("T")
R = TypeVar("R")

# Decorator: require that login has been done (access_token present)
def requires_auth(func: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
    @wraps(func)
    async def wrapper(ctx: Context, *args, **kwargs) -> Dict[str, Any]:
        superset_ctx: SupersetContext = ctx.request_context.lifespan_context
        if not superset_ctx.access_token:
            return {"error": "Not authenticated. Please call superset_auth_login first."}
        # ensure Authorization header is set
        superset_ctx.client.headers.update({"Authorization": f"Bearer {superset_ctx.access_token}"})
        return await func(ctx, *args, **kwargs)
    return wrapper

# Decorator: catch exceptions and return structured error
def handle_api_errors(func: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
    @wraps(func)
    async def wrapper(ctx: Context, *args, **kwargs) -> Dict[str, Any]:
        try:
            return await func(ctx, *args, **kwargs)
        except Exception as e:
            return {"error": f"Error in {func.__name__}: {str(e)}"}
    return wrapper

# Fetch CSRF token for mutating API calls
def get_csrf_token(ctx: Context) -> Awaitable[Optional[str]]:
    async def _inner() -> Optional[str]:
        superset_ctx: SupersetContext = ctx.request_context.lifespan_context
        resp = await superset_ctx.client.get("/api/v1/security/csrf_token/")
        if resp.status_code == 200:
            token = resp.json().get("result")
            superset_ctx.csrf_token = token
            return token
        return None
    return _inner()

# Generic helper for GET/POST/PUT/DELETE to Superset API
async def make_api_request(
    ctx: Context,
    method: str,
    endpoint: str,
    data: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    client = superset_ctx.client
    headers: Dict[str, str] = {}

    # Always add X-CSRFToken for mutating requests (POST, PUT, DELETE)
    if method.lower() in ("post", "put", "delete"):
        token = superset_ctx.csrf_token or await get_csrf_token(ctx)
        logging.info(f"CSRF token: {token}")
        if token:
            headers["X-CSRFToken"] = token
    logging.info(f"Headers: {headers}")
    # perform the HTTP method
    if method.lower() == "get":
        resp = await client.get(endpoint, params=params)
    elif method.lower() == "post":
        resp = await client.post(endpoint, json=data, params=params, headers=headers)
    elif method.lower() == "put":
        resp = await client.put(endpoint, json=data, headers=headers)
    elif method.lower() == "delete":
        resp = await client.delete(endpoint, headers=headers)
    else:
        return {"error": f"Unsupported HTTP method: {method}"}

    if resp.status_code not in (200, 201):
        return {"error": f"API request failed: {resp.status_code} – {resp.text}"}
    return resp.json()

# Tool: authenticate via API login and store access token
@mcp.tool()
@handle_api_errors
async def superset_auth_login(
    ctx: Context,
    username: Optional[str] = None,
    password: Optional[str] = None,
    refresh: bool = False
) -> Dict[str, Any]:
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    username = username or SUPERSET_USERNAME
    password = password or SUPERSET_PASSWORD
    if not username or not password:
        return {"error": "Username and password must be provided"}

    # single, un-duplicated login flow
    body = {"username": username, "password": password, "provider": "db", "refresh": refresh}
    resp = await superset_ctx.client.post("/api/v1/security/login", json=body)
    if resp.status_code != 200:
        return {"error": f"Login failed: {resp.status_code} – {resp.text}"}

    # pull the JWT out of whatever shape Superset gives you
    payload = resp.json()
    token = payload.get("access_token") or payload.get("result", {}).get("access_token")
    if not token:
        return {"error": "Login failed: no access_token returned"}

    superset_ctx.access_token = token
    logging.info(f"Access token: {token}")
    superset_ctx.client.headers.update({"Authorization": f"Bearer {token}"})
    return {"message": "Successfully authenticated with Superset"}


# Tool: logout (clear token)
@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_auth_logout(ctx: Context) -> Dict[str, Any]:
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    # optional revoke
    await superset_ctx.client.post("/api/v1/security/logout")
    superset_ctx.access_token = None
    superset_ctx.csrf_token = None
    return {"message": "Successfully logged out from Superset"}

# Tool: list dashboards
@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_list(ctx: Context) -> Dict[str, Any]:
    return await make_api_request(ctx, "get", "/api/v1/dashboard/")

# Tool: list charts
@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_list(ctx: Context) -> Dict[str, Any]:
    return await make_api_request(ctx, "get", "/api/v1/chart/")

# Tool: create a chart in Superset
@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_create(
    ctx: Context,
    slice_name: str,
    datasource_id: int,
    datasource_type: str,
    viz_type: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a new chart in Superset.
    Args:
        slice_name: Name/title of the chart
        datasource_id: ID of the dataset or SQL table
        datasource_type: Type of datasource ('table' for datasets, 'query' for SQL)
        viz_type: Visualization type (e.g., 'bar', 'line', 'pie', etc.)
        params: Visualization parameters (dict)
    Returns:
        A dictionary with the created chart information including its ID
    """
    payload = {
        "slice_name": slice_name,
        "datasource_id": datasource_id,
        "datasource_type": datasource_type,
        "viz_type": viz_type,
        "params": params,
    }
    return await make_api_request(ctx, "post", "/api/v1/chart/", data=payload)

# Entry point
if __name__ == "__main__":
    print("Starting Superset MCP server...")
    mcp.run(transport="stdio")
