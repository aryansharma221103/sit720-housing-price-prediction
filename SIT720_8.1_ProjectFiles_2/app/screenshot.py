import time
from playwright.sync_api import sync_playwright

OUT_DIR = "../figures"

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 1400})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    time.sleep(3)

    page.screenshot(path=f"{OUT_DIR}/app_screenshot_1_form.png", full_page=True)
    print("Saved form screenshot")

    # Fill in a few fields to make a realistic Mosman house example, then predict
    page.get_by_role("combobox", name="Suburb", exact=True).click()
    page.get_by_role("option", name="Mosman", exact=True).click()
    page.get_by_role("combobox", name="Property type", exact=True).click()
    page.get_by_role("option", name="House", exact=True).click()

    page.get_by_role("button", name="Predict sale price").click()
    result_locator = page.locator("text=Predicted sale price")
    result_locator.wait_for(timeout=15000)
    result_locator.scroll_into_view_if_needed()
    time.sleep(1)

    page.screenshot(path=f"{OUT_DIR}/app_screenshot_2_prediction.png", full_page=True)
    print("Saved prediction screenshot")

    # Also grab a focused viewport shot centred on the result for the report
    page.screenshot(path=f"{OUT_DIR}/app_screenshot_2b_prediction_zoom.png", full_page=False)
    print("Saved zoomed prediction screenshot")

    # Batch tab screenshot for the report's usage instructions
    page.get_by_role("tab", name="Upload a CSV (batch)").click()
    time.sleep(1)
    page.screenshot(path=f"{OUT_DIR}/app_screenshot_3_batch_tab.png", full_page=True)
    print("Saved batch tab screenshot")

    browser.close()
