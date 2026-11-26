import os
import io
import cv2
import qrcode
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

# ---------------------------------------------------------
# 1. Excel Configuration & Columns
# ---------------------------------------------------------
EXCEL_FILE = "TL weighment details.xlsx"

COLUMNS = [
    "Transporter Name",
    "Date",
    "Vehicle number",
    "Ship to party",
    "Material",
    "Qty",
    "Temp observed",
    "Tare weight",
    "Second weight",
    "Density @ 29.5",
    "Qty @L29 as per invoice",
    "Time of filling",
    "Date of filling",
]


class WeighmentExtraction(BaseModel):
    date: str = Field(description="Date of weighment or invoice (e.g., DD/MM/YYYY)")
    vehicle_number: str = Field(description="Tank Lorry / Vehicle registration number")
    ship_to_party: str = Field(description="Ship to party / Consignee / Customer name and location")
    material: str = Field(description="Material or product name/code (e.g., MS, HSD, FO, Bitumen)")
    qty: str = Field(description="Quantity (Net weight or billed quantity as stated)")
    temp_observed: str = Field(description="Observed temperature (e.g., in °C)")
    tare_weight: str = Field(description="Tare weight / First weight of the vehicle")
    second_weight: str = Field(description="Second weight / Gross weight of the vehicle")
    density_at_29_5: str = Field(description="Density @ 29.5°C (or @ 29.5)")
    qty_at_l29_invoice: str = Field(description="Qty @ L29 (or KL/Ltrs at 29.5°C) as per invoice")
    time_of_filling: str = Field(description="Time of filling / loading / weighment (e.g., HH:MM)")
    date_of_filling: str = Field(description="Date of filling / loading (e.g., DD/MM/YYYY)")


# ---------------------------------------------------------
# 2. HD & Contrast Enhancement Pipeline (OpenCV)
# ---------------------------------------------------------
def enhance_photo_to_hd(pil_img: Image.Image, clip_limit: float = 2.5, sharpen_strength: float = 1.5) -> Image.Image:
    """Normalizes resolution to crisp 2K HD, boosts contrast with CLAHE, and sharpens text."""
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1. Scale image to crisp HD (2200px on longest side)
    h, w = img_bgr.shape[:2]
    target_edge = 2200
    if max(h, w) < target_edge:
        scale = target_edge / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    elif max(h, w) > 2600:
        scale = target_edge / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. Edge-preserving noise reduction
    denoised = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=40, sigmaSpace=40)

    # 3. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    contrast_bgr = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    # 4. Unsharp masking for crisp digits and letters
    gaussian = cv2.GaussianBlur(contrast_bgr, (0, 0), sigmaX=2.0)
    sharpened_bgr = cv2.addWeighted(contrast_bgr, 1.0 + sharpen_strength, gaussian, -sharpen_strength, 0)

    return Image.fromarray(cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------
# 3. Gemini Vision Multi-Image Extraction
# ---------------------------------------------------------
def extract_details_from_images(invoice_img: Image.Image, slip_img: Image.Image, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)

    prompt = (
        "You are an expert operations data extractor for Tank Lorry (TL) invoices and weighment slips. "
        "Image 1 is the Invoice / Delivery Challan. Image 2 is the Weighment Slip. "
        "Carefully extract the following fields across both enhanced HD images:\n"
        "1. Date\n"
        "2. Vehicle number\n"
        "3. Ship to party\n"
        "4. Material\n"
        "5. Qty\n"
        "6. Temp observed\n"
        "7. Tare weight\n"
        "8. Second weight\n"
        "9. Density @ 29.5\n"
        "10. Qty @L29 as per invoice\n"
        "11. Time of filling\n"
        "12. Date of filling\n\n"
        "Return clean values with exact numerical precision. If a field is not visible in either image, return an empty string."
    )

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[invoice_img, slip_img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=WeighmentExtraction,
        ),
    )

    parsed: WeighmentExtraction = response.parsed
    return {
        "Date": parsed.date,
        "Vehicle number": parsed.vehicle_number,
        "Ship to party": parsed.ship_to_party,
        "Material": parsed.material,
        "Qty": parsed.qty,
        "Temp observed": parsed.temp_observed,
        "Tare weight": parsed.tare_weight,
        "Second weight": parsed.second_weight,
        "Density @ 29.5": parsed.density_at_29_5,
        "Qty @L29 as per invoice": parsed.qty_at_l29_invoice,
        "Time of filling": parsed.time_of_filling,
        "Date of filling": parsed.date_of_filling,
    }



# ---------------------------------------------------------
# 4. Excel File Handler
# ---------------------------------------------------------
def init_or_load_excel(filepath: str):
    if not os.path.exists(filepath):
        wb = Workbook()
        ws = wb.active
        ws.title = "TL Weighment"
        ws.append(COLUMNS)

        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        for col_idx in range(1, len(COLUMNS) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            ws.column_dimensions[cell.column_letter].width = 22

        wb.save(filepath)


def append_row_to_excel(filepath: str, row_dict: dict):
    init_or_load_excel(filepath)
    wb = load_workbook(filepath)
    ws = wb.active
    ws.append([row_dict.get(col, "") for col in COLUMNS])
    wb.save(filepath)


# ---------------------------------------------------------
# 5. Mobile-Friendly Web Portal UI
# ---------------------------------------------------------
st.set_page_config(page_title="TL Weighment Portal", page_icon="🚛", layout="centered")

try:
    API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
except Exception:
    API_KEY = os.environ.get("GEMINI_API_KEY", "")

st.title("🚛 TL Weighment Entry Portal")

tab_entry, tab_admin, tab_qr = st.tabs(["📝 New Entry", "📊 Excel Sheet & Download", "📱 QR Code"])

with tab_entry:
    st.markdown("Enter the transporter name and upload photos of the Invoice and Weighment Slip.")

    transporter_name = st.text_input("1. Enter Transporter Name *", placeholder="e.g., ABC Transport")

    st.markdown("##### 2. Upload Required Photos")
    invoice_file = st.file_uploader("📄 Upload Photo of Invoice *", type=["jpg", "jpeg", "png", "webp"])
    slip_file = st.file_uploader("⚖️ Upload Photo of Weighment Slip *", type=["jpg", "jpeg", "png", "webp"])

    if invoice_file and slip_file:
        if st.button("✨ Enhance Photos to HD & Extract Details", type="primary", use_container_width=True):
            if not transporter_name.strip():
                st.warning("Please enter the Transporter Name first.")
            elif not API_KEY:
                st.error("API Key missing! Please add GEMINI_API_KEY in your Streamlit App Settings (Secrets).")
            else:
                with st.spinner("Enhancing contrast, converting to HD, and capturing fields..."):
                    raw_inv = Image.open(invoice_file)
                    raw_slip = Image.open(slip_file)

                    hd_inv = enhance_photo_to_hd(raw_inv)
                    hd_slip = enhance_photo_to_hd(raw_slip)

                    st.session_state["hd_inv"] = hd_inv
                    st.session_state["hd_slip"] = hd_slip

                    try:
                        extracted = extract_details_from_images(hd_inv, hd_slip, API_KEY)
                        extracted["Transporter Name"] = transporter_name.strip()
                        st.session_state["extracted_data"] = extracted
                        st.success("Photos converted to HD and details captured! Please verify below.")
                    except Exception as e:
                        st.error(f"Extraction error: {e}")

    if "extracted_data" in st.session_state:
        with st.expander("🔍 View Enhanced HD Photos"):
            c1, c2 = st.columns(2)
            with c1:
                st.image(st.session_state["hd_inv"], caption="Contrast-Enhanced HD Invoice", use_container_width=True)
            with c2:
                st.image(st.session_state["hd_slip"], caption="Contrast-Enhanced HD Weighment Slip", use_container_width=True)

        st.subheader("3. Verify & Update Excel Sheet")
        data = st.session_state["extracted_data"]

        with st.form("confirm_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                v_transporter = st.text_input("Transporter Name", value=data.get("Transporter Name", ""))
                v_date = st.text_input("Date", value=data.get("Date", ""))
                v_vehicle = st.text_input("Vehicle number", value=data.get("Vehicle number", ""))
                v_party = st.text_input("Ship to party", value=data.get("Ship to party", ""))
                v_material = st.text_input("Material", value=data.get("Material", ""))
                v_qty = st.text_input("Qty", value=data.get("Qty", ""))
                v_temp = st.text_input("Temp observed", value=data.get("Temp observed", ""))
            with col_b:
                v_tare = st.text_input("Tare weight", value=data.get("Tare weight", ""))
                v_second = st.text_input("Second weight", value=data.get("Second weight", ""))
                v_density = st.text_input("Density @ 29.5", value=data.get("Density @ 29.5", ""))
                v_qty_l29 = st.text_input("Qty @L29 as per invoice", value=data.get("Qty @L29 as per invoice", ""))
                v_time_fill = st.text_input("Time of filling", value=data.get("Time of filling", ""))
                v_date_fill = st.text_input("Date of filling", value=data.get("Date of filling", ""))

            save_btn = st.form_submit_button("✅ Save Record to Excel Sheet", use_container_width=True)
            if save_btn:
                row_to_save = {
                    "Transporter Name": v_transporter,
                    "Date": v_date,
                    "Vehicle number": v_vehicle,
                    "Ship to party": v_party,
                    "Material": v_material,
                    "Qty": v_qty,
                    "Temp observed": v_temp,
                    "Tare weight": v_tare,
                    "Second weight": v_second,
                    "Density @ 29.5": v_density,
                    "Qty @L29 as per invoice": v_qty_l29,
                    "Time of filling": v_time_fill,
                    "Date of filling": v_date_fill,
                }
                append_row_to_excel(EXCEL_FILE, row_to_save)
                st.success("✅ Record saved to `TL weighment details.xlsx`!")
                del st.session_state["extracted_data"]

with tab_admin:
    st.subheader("📊 Master Excel Sheet (`TL weighment details.xlsx`)")
    init_or_load_excel(EXCEL_FILE)
    df = pd.read_excel(EXCEL_FILE)
    st.dataframe(df, use_container_width=True)

    with open(EXCEL_FILE, "rb") as f:
        st.download_button(
            label="📥 Download Updated TL weighment details.xlsx",
            data=f,
            file_name=EXCEL_FILE,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with st.expander("📤 Upload Existing Excel File (Optional)"):
        st.caption("If you already have a 'TL weighment details.xlsx' file on your computer, upload it here so new entries append to it.")
        uploaded_xlsx = st.file_uploader("Select Excel file", type=["xlsx"])
        if uploaded_xlsx is not None:
            with open(EXCEL_FILE, "wb") as f:
                f.write(uploaded_xlsx.getbuffer())
            st.success("Existing Excel sheet loaded! Refresh or switch tabs to view.")

with tab_qr:
    st.subheader("📱 Generate Printable QR Code")
    app_url = st.text_input("Paste your live Streamlit App URL here:", placeholder="https://your-app-name.streamlit.app")
    if app_url:
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(app_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        st.image(buf.getvalue(), caption="Scan to open TL Weighment Portal", width=280)
        st.download_button("⬇️ Download QR Code PNG", data=buf.getvalue(), file_name="TL_Weighment_QR.png", mime="image/png")
