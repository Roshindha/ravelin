"""The review screenshots of the arena (B24), taken in real Chrome through Playwright (``pip install playwright``; a system Chrome is used).

    .venv/bin/python scripts/take_screenshots.py            # starts the app (RAVELIN_DEPLOY=1) on a free port and writes docs/screenshots/*.png

Every game state comes from the REAL engine, driven by real clicks / taps / key presses on the placement board and by the app's own ``?demo=`` query parameters
(``place`` | ``battle`` | ``report`` | ``held``, plus ``&tick=N``; ``?mode=watch&pos=N``). The human script is the row (mm) script (sniper x4 and bulwark x2 on border tiles 0-5).
Nothing here touches game logic; human runs are not logged in deployed mode. Desktop shots are 1500 px wide, mobile shots 390 px wide.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    from playwright.sync_api import sync_playwright
    port = free_port()
    env = {**os.environ, "RAVELIN_DEPLOY": "1"}
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"), "--server.port", str(port), "--server.headless", "true",
                            "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{port}/"
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(url, timeout=2)
                break
            except OSError:
                time.sleep(1)
        OUT.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome")

            def page(w=1500, h=1900, **kw):
                return b.new_context(viewport={"width": w, "height": h}, **kw).new_page()

            def shot(pg, name):
                pg.mouse.move(2, 2)
                pg.screenshot(path=str(OUT / name))

            pg = page(); pg.goto(url); pg.wait_for_selector(".pl"); pg.wait_for_timeout(600)           # 1 placing: keys, taps, Shift+tap
            pg.keyboard.press("2")
            for t in range(4):
                pg.click(f'rect.hit[data-b="{t}"]')
            pg.keyboard.press("3")
            for t in (4, 5):
                pg.click(f'rect.hit[data-b="{t}"]')
            pg.keyboard.press("1"); pg.click('rect.hit[data-b="20"]', modifiers=["Shift"])
            shot(pg, "ravelin_1_placing_green_tiles_and_ghosts.png")
            pg = page(); pg.goto(url + "?demo=battle&tick=8"); pg.wait_for_selector("iframe"); pg.wait_for_timeout(2500)       # 2 step one tick: numbers on the board
            pg.frame_locator("iframe").first.locator("#step").click(); pg.wait_for_timeout(120)
            shot(pg, "ravelin_2_battle_hp_bars_rings_and_damage_numbers.png")
            pg = page(); pg.goto(url + "?demo=battle"); pg.wait_for_selector("iframe"); pg.wait_for_timeout(2600)                # 3 autoplay stops at the first key moment
            shot(pg, "ravelin_3_paused_on_a_key_moment_with_continue.png")
            fr = pg.frame_locator("iframe").first; fr.locator("#rng").check(); fr.locator("#pth").check(); pg.wait_for_timeout(200)
            shot(pg, "ravelin_4_range_and_path_overlays.png")
            pg = page(); pg.goto(url + "?demo=battle&tick=32"); pg.wait_for_selector("iframe"); pg.wait_for_timeout(2000)
            shot(pg, "ravelin_5_end_sticker_BREACH_at_the_core_tick.png")
            pg = page(1500, 2500); pg.goto(url + "?demo=report"); pg.wait_for_selector("iframe"); pg.wait_for_timeout(3500)
            shot(pg, "ravelin_6_report_cards_and_watch_buttons.png")
            pg = page(); pg.goto(url + "?mode=watch&pos=2"); pg.wait_for_selector(".bl-board svg", timeout=40000); pg.wait_for_timeout(1500)
            shot(pg, "ravelin_7_watch_the_ai_paused_at_a_decision.png")
            pg = page(); pg.goto(url + "?mode=watch&pos=99&tick=6"); pg.wait_for_selector("iframe", timeout=40000); pg.wait_for_timeout(2500)
            shot(pg, "ravelin_8_watch_the_ai_battle_agents_army.png")
            pg = page(390, 1900, device_scale_factor=1, has_touch=True, is_mobile=True); pg.goto(url); pg.wait_for_selector(".pl"); pg.wait_for_timeout(800)
            pg.tap('.card[data-u="1"]')
            for t in (0, 1, 2):
                pg.tap(f'rect.hit[data-b="{t}"]')
            shot(pg, "ravelin_9_mobile_390_placing.png")
            pg = page(390, 1900, device_scale_factor=1, has_touch=True, is_mobile=True); pg.goto(url + "?demo=battle&tick=8"); pg.wait_for_selector("iframe"); pg.wait_for_timeout(2500)
            pg.frame_locator("iframe").first.locator("#step").tap(); pg.wait_for_timeout(120)
            shot(pg, "ravelin_10_mobile_390_battle.png")
            b.close()
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
