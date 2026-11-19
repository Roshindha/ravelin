"""B18e-b: the review screenshots, taken in real Chrome through the DevTools protocol (docs/SPEC_V2.md, Gate changelog row (oo) (g)).

    .venv/bin/python scripts/take_screenshots.py            # starts the app on a free port and writes docs/screenshots/*.png

Rules (row (oo) (g)): 1500x1000, default entry point ``app/streamlit_app.py``, named by what they show. The human script is the row (mm) script (buy sniper x4, bulwark x2,
END BUY, deploy in the order bought on the lowest legal tiles 0-5); HELD = one Scout at (0,3); Watch = the first deploy step with the heatmap on; one mobile shot at
390x844. Every game state comes from the REAL engine, driven by the app's own ``?demo=`` query parameters (see app/streamlit_app.py) and by real mouse / touch events
(the hover and the click-to-deploy shots use ``Input.dispatchMouseEvent``). Nothing here touches game logic. Human runs go to a temp file, never to results_v2.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
W, H = 1500, 1000


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Chrome:
    """A headless Chrome with one tab, driven over the DevTools websocket (synchronous wrapper around asyncio)."""

    def __init__(self):
        self.port, self.dir = free_port(), tempfile.mkdtemp(prefix="bl_chrome_")
        self.proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={self.port}", f"--user-data-dir={self.dir}", "--no-first-run",
                                      "--hide-scrollbars", f"--window-size={W},{H}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json"))
                self.ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("Chrome did not start")
        self.loop = asyncio.new_event_loop()
        self.ws = self.loop.run_until_complete(websockets.connect(self.ws_url, max_size=64 * 2 ** 20))
        self.n, self.events = 0, []
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("Network.enable")

    def send(self, method: str, **params):
        self.n += 1
        mid = self.n

        async def go():
            await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
            while True:
                msg = json.loads(await self.ws.recv())
                if msg.get("id") == mid:
                    if "error" in msg:
                        raise RuntimeError(f"{method}: {msg['error']}")
                    return msg.get("result", {})
                self.events.append(msg)
        return self.loop.run_until_complete(go())


    def viewport(self, w: int, h: int, mobile: bool = False):
        self.send("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=mobile)
        if mobile:
            self.send("Emulation.setTouchEmulationEnabled", enabled=True)

    def go(self, url: str):
        self.send("Page.navigate", url=url)

    def js(self, expr: str):
        r = self.send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(str(r["exceptionDetails"])[:400])
        return r["result"].get("value")

    def wait(self, expr: str, timeout: float = 90):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.js(expr):
                    return True
            except RuntimeError:
                pass
            time.sleep(0.4)
        raise TimeoutError(expr)

    def shot(self, path: Path):
        data = self.send("Page.captureScreenshot", format="png", captureBeyondViewport=False)["data"]
        path.write_bytes(base64.b64decode(data))
        print("wrote", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)

    def mouse(self, kind: str, x: float, y: float, **kw):
        self.send("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1, **kw)

    def click(self, x: float, y: float):
        self.mouse("mouseMoved", x, y)
        self.mouse("mousePressed", x, y)
        self.mouse("mouseReleased", x, y)


    def close(self):
        try:
            self.loop.run_until_complete(self.ws.close())
        finally:
            self.proc.terminate()


# ----------------------------------------------------------------------------------------------------------------------------- the app under test
def start_app(runs_path: str):
    port = free_port()
    env = {**os.environ, "BREACHLENS_DUEL_RUNS": runs_path}
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"), "--server.port", str(port), "--server.headless", "true",
                             "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health")
            return proc, f"http://127.0.0.1:{port}/"
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("streamlit did not start")


READY = "document.querySelector('[data-testid=stApp]') && !document.querySelector('[data-testid=stStatusWidget]') && document.body.innerText.includes('Ravelin')"


def load(c: Chrome, url: str, settle: float = 1.5):
    c.go(url)
    c.wait(READY)
    time.sleep(settle)


def tile_xy(c: Chrome, b: int) -> tuple:
    """Page coordinates (centre) of border tile b's hit rectangle inside the board component (shadow DOM)."""
    return tuple(c.js("""(() => { const find = (root) => { const r = root.querySelector && root.querySelector('rect.hit[data-b="%d"]'); if (r) return r;
      for (const e of root.querySelectorAll('*')) { if (e.shadowRoot) { const f = find(e.shadowRoot); if (f) return f; } } return null; };
      const r = find(document); const b = r.getBoundingClientRect(); return [b.x + b.width / 2, b.y + b.height / 2]; })()""" % b))


def scroll_to(c: Chrome, key: str = "card_board") -> None:
    """Scroll so that the keyed card ``st-key-<key>`` starts near the top of the viewport."""
    c.js("""(() => { const e = document.querySelector("[class*='st-key-%s']"); const m = document.querySelector('[data-testid=stMain]') || document.scrollingElement;
      m.scrollTop += e.getBoundingClientRect().top - 12; })()""" % key)
    time.sleep(0.4)


def scroll_bottom(c: Chrome) -> None:
    c.js("(() => { const m = document.querySelector('[data-testid=stMain]') || document.scrollingElement; m.scrollTop = m.scrollHeight; })()")
    time.sleep(0.4)


def main(out: Path = OUT) -> None:
    global OUT
    OUT = out
    OUT.mkdir(parents=True, exist_ok=True)
    runs = os.path.join(tempfile.mkdtemp(prefix="bl_runs_"), "runs.jsonl")
    app, base = start_app(runs)
    c = Chrome()
    try:
        c.viewport(W, H)
        load(c, base + "?demo=buy")
        c.shot(OUT / "ravelin_1a_title_top_two_units_bought.png")
        scroll_bottom(c)
        c.shot(OUT / "ravelin_1b_shop_two_units_bought.png")
        load(c, base + "?demo=deploy")
        scroll_to(c)
        x, y = tile_xy(c, 5)
        c.mouse("mouseMoved", x, y)
        time.sleep(0.6)
        c.shot(OUT / "ravelin_2_deploy_hovering_a_free_tile.png")
        load(c, base + "?demo=battle&tick=16")
        scroll_to(c)
        c.shot(OUT / "ravelin_3_battle_midpoint_tick_16.png")
        load(c, base + "?demo=battle&tick=32")
        scroll_to(c)
        c.shot(OUT / "ravelin_4_end_sticker_BREACH_row_mm_play.png")
        load(c, base + "?demo=held&tick=1")
        scroll_to(c)
        c.shot(OUT / "ravelin_5_end_sticker_HELD_one_scout_at_0_3.png")
        load(c, base + "?demo=report")
        scroll_to(c, "grid_report")
        c.shot(OUT / "ravelin_6_report_cards.png")
        load(c, base + "?mode=watch&pos=9", settle=3)
        c.wait("document.body.innerText.includes('Heatmap overlay')")
        scroll_to(c)
        c.shot(OUT / "ravelin_7_watch_first_deploy_step_heatmap_on.png")
        c.viewport(390, 844, mobile=True)
        load(c, base + "?demo=deploy")
        scroll_to(c, "card_board")
        c.shot(OUT / "ravelin_8_mobile_390x844_deploy.png")
    finally:
        c.close()
        app.terminate()


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
