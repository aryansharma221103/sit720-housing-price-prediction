from pathlib import Path
from playwright.sync_api import sync_playwright

HTML = Path(__file__).parent / "report.html"
PDF = Path(__file__).parent / "SIT720_8.1_Report.pdf"

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    page = browser.new_page()
    page.goto(HTML.as_uri(), wait_until="networkidle", timeout=60000)
    page.pdf(
        path=str(PDF),
        format="A4",
        print_background=True,
        margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
    )
    browser.close()

print(f"Wrote {PDF} ({PDF.stat().st_size:,} bytes)")
