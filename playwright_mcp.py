#!/usr/bin/env python3
"""
Simple Playwright MCP Server
A minimal MCP server for Playwright browser automation
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolResult,
    ListToolsResult,
)

# Global browser instance
browser: Optional[Browser] = None
playwright_instance = None

app = Server("playwright-server")

async def get_browser():
    """Get or create browser instance"""
    global browser, playwright_instance
    if browser is None:
        playwright_instance = await async_playwright().start()
        browser = await playwright_instance.chromium.launch(headless=False)
    return browser

async def cleanup_browser():
    """Clean up browser resources"""
    global browser, playwright_instance
    if browser:
        await browser.close()
        browser = None
    if playwright_instance:
        await playwright_instance.stop()
        playwright_instance = None

@app.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="launch_browser",
            description="Launch a browser instance",
            inputSchema={
                "type": "object",
                "properties": {
                    "headless": {
                        "type": "boolean",
                        "description": "Whether to run browser in headless mode",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="navigate_to",
            description="Navigate to a URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="take_screenshot",
            description="Take a screenshot of the current page",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to save screenshot (optional)"
                    }
                }
            }
        ),
        Tool(
            name="get_page_title",
            description="Get the title of the current page",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="click_element",
            description="Click an element on the page",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of element to click"
                    }
                },
                "required": ["selector"]
            }
        ),
        Tool(
            name="type_text",
            description="Type text into an element",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of element to type into"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type"
                    }
                },
                "required": ["selector", "text"]
            }
        ),
        Tool(
            name="get_page_content",
            description="Get the HTML content of the current page",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="wait_for_element",
            description="Wait for an element to appear on the page",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of element to wait for"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in milliseconds (default: 30000)",
                        "default": 30000
                    }
                },
                "required": ["selector"]
            }
        ),
        Tool(
            name="close_browser",
            description="Close the browser instance",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

# Global page instance
current_page: Optional[Page] = None

@app.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    global current_page

    try:
        if name == "launch_browser":
            await get_browser()
            context = await browser.new_context()
            current_page = await context.new_page()
            return [TextContent(type="text", text="Browser launched successfully")]

        elif name == "navigate_to":
            if not current_page:
                await get_browser()
                context = await browser.new_context()
                current_page = await context.new_page()

            url = arguments["url"]
            await current_page.goto(url)
            return [TextContent(type="text", text=f"Navigated to {url}")]

        elif name == "take_screenshot":
            if not current_page:
                return [TextContent(type="text", text="No active page. Please launch browser first.")]

            path = arguments.get("path")
            screenshot = await current_page.screenshot(path=path)

            if path:
                return [TextContent(type="text", text=f"Screenshot saved to {path}")]
            else:
                # Return base64 encoded screenshot
                import base64
                screenshot_b64 = base64.b64encode(screenshot).decode()
                return [TextContent(type="text", text=f"data:image/png;base64,{screenshot_b64}")]

        elif name == "get_page_title":
            if not current_page:
                return [TextContent(type="text", text="No active page. Please launch browser first.")]

            title = await current_page.title()
            return [TextContent(type="text", text=f"Page title: {title}")]

        elif name == "click_element":
            if not current_page:
                return [TextContent(type="text", text="No active page. Please launch browser first.")]

            selector = arguments["selector"]
            await current_page.click(selector)
            return [TextContent(type="text", text=f"Clicked element: {selector}")]

        elif name == "type_text":
            if not current_page:
                return [TextContent(type="text", text="No active page. Please launch browser first.")]

            selector = arguments["selector"]
            text = arguments["text"]
            await current_page.fill(selector, text)
            return [TextContent(type="text", text=f"Typed text into {selector}")]

        elif name == "get_page_content":
            if not current_page:
                return [TextContent(type="text", text="No active page. Please launch browser first.")]

            content = await current_page.content()
            return [TextContent(type="text", text=content)]

        elif name == "wait_for_element":
            if not current_page:
                return [TextContent(type="text", text="No active page. Please launch browser first.")]

            selector = arguments["selector"]
            timeout = arguments.get("timeout", 30000)
            await current_page.wait_for_selector(selector, timeout=timeout)
            return [TextContent(type="text", text=f"Element found: {selector}")]

        elif name == "close_browser":
            await cleanup_browser()
            current_page = None
            return [TextContent(type="text", text="Browser closed")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Main entry point"""
    # Use stdio server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="playwright-server",
                server_version="1.0.0",
                capabilities={
                    "tools": {},
                },
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())