import streamlit as st
import ee
import requests
import pandas as pd
import datetime

# --- CONFIGURATION FOR CHOORALMALA DISASTER ---
PROJECT_ID = 'tokyo-mark-458613-p1'  # Keep your project ID
LOCATION_NAME = "Chooralmala, Wayanad (HISTORY: July 30 2024)"
LAT = 11.54
LON = 76.13

# TARGET DATES (The Disaster Window)
TARGET_DATE = "2024-07-30"
START_DATE = "2024-07-20"


# --- 1. AUTHENTICATE ---
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize(project=PROJECT_ID)
        return True
    except:
        try:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID)
            return True
        except Exception as e:
            st.error(f"Auth Failed: {e}")
            return False


# --- 2. FETCH HISTORICAL RAIN (Archive API) ---
def get_historical_rain(lat, lon, start, end):
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&hourly=rain"
        response = requests.get(url)
        data = response.json()

        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])

        total_rain = df['rain'].sum()
        max_intensity = df['rain'].max()

        return total_rain, max_intensity, df
    except Exception as e:
        st.error(f"Weather Archive Error: {e}")
        return 0, 0, pd.DataFrame()


# --- 3. FETCH SENTINEL-1 (Structural Radar) ---
def get_sentinel_stability(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(500)

        # Look for images in July 2024
        collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                      .filterBounds(roi)
                      .filterDate('2024-07-01', '2024-07-29')
                      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                      .filter(ee.Filter.eq('instrumentMode', 'IW'))
                      .select('VV'))

        count = collection.size().getInfo()

        if count < 2:
            return "Data Gap (Sentinel-1 Blind)", 0.0

        img_late = collection.sort('system:time_start', False).first()
        img_early = collection.sort('system:time_start', True).first()

        # Get date of the last successful pass
        last_date = ee.Date(img_late.get('system:time_start')).format('YYYY-MM-dd').getInfo()

        diff = img_late.subtract(img_early).abs()
        change_score = diff.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        ).getInfo()['VV']

        return f"Last Pass: {last_date}", change_score

    except Exception as e:
        return f"Sentinel Error: {e}", 0.0


# --- 4. FETCH NASA SMAP (Soil Moisture - The Filler) ---
def get_smap_moisture(lat, lon, date):
    try:
        point = ee.Geometry.Point([lon, lat])

        # SMAP Level-3 Daily Soil Moisture (9km resolution)
        # We look at the 3 days leading up to the target date
        start_look = (pd.to_datetime(date) - pd.Timedelta(days=3)).strftime('%Y-%m-%d')

        dataset = (ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006")
                   .filterBounds(point)
                   .filterDate(start_look, date)
                   .select('soil_moisture_am'))  # Morning pass is better for soil

        count = dataset.size().getInfo()

        if count == 0:
            return "No SMAP Data", 0.0

        # Get the average moisture over the last 3 days
        mean_img = dataset.mean()

        # Reduce to get the value at our point
        moisture_val = mean_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=9000  # 9km resolution
        ).getInfo()['soil_moisture_am']

        # SMAP returns volumetric water content (cm3/cm3).
        # > 0.40 is typically very wet/saturated for soil.
        return "Data Found", moisture_val

    except Exception as e:
        return f"SMAP Error: {e}", 0.0


# --- 5. THE REPORT CARD ---
def main():
    st.set_page_config(page_title="Forensic Analysis", page_icon="🕵️")

    st.title("🕵️ Forensic Analysis: Chooralmala")
    st.warning(f"Simulating Data for: {TARGET_DATE}")

    initialize_earth_engine()

    # 1. Weather
    rain_total, rain_peak, rain_df = get_historical_rain(LAT, LON, START_DATE, TARGET_DATE)

    # 2. Sentinel-1 (Structure)
    sentinel_status, sentinel_score = get_sentinel_stability(LAT, LON)

    # 3. NASA SMAP (Moisture)
    smap_status, smap_score = get_smap_moisture(LAT, LON, TARGET_DATE)

    # VISUALIZE
    col1, col2, col3 = st.columns(3)

    col1.metric("🌧️ Rain (10 Days)", f"{rain_total:.1f} mm")
    col2.metric("📡 Sentinel-1 Instability", f"{sentinel_score:.2f}", help="Last pass date")
    col3.metric("💧 SMAP Soil Moisture", f"{smap_score:.3f}", help=">0.4 is Saturated")

    # --- THE HYBRID DECISION ENGINE ---
    alert_level = "GREEN"
    alert_color = "success"
    reasons = []

    # Rule 1: Structural Failure (Sentinel)
    if sentinel_score > 2.0:
        reasons.append("Ground Shift Detected (Sentinel-1)")

    # Rule 2: Soil Saturation (SMAP)
    # 0.45 cm3/cm3 is near porosity limit for many soils
    if isinstance(smap_score, float) and smap_score > 0.4:
        reasons.append(f"Soil Saturated ({smap_score:.2f} moisture)")

    # Rule 3: Heavy Rain Trigger
    if rain_total > 200:
        reasons.append(f"Extreme Rainfall ({rain_total}mm)")

    # LOGIC:
    if len(reasons) >= 2:
        alert_level = "RED ALERT (EVACUATE)"
        alert_color = "error"
    elif len(reasons) == 1:
        alert_level = "YELLOW ALERT (PREPARE)"
        alert_color = "warning"

    if alert_level == "GREEN":
        st.success("✅ STATUS: SAFE")
    elif alert_level.startswith("YELLOW"):
        st.warning(f"⚠️ STATUS: {alert_level}")
        st.write("Risk Factors: " + ", ".join(reasons))
    else:
        st.error(f"🚨 STATUS: {alert_level}")
        st.write("**CRITICAL TRIGGERS:**")
        for r in reasons:
            st.markdown(f"- {r}")

    st.markdown("### 🌧️ Rainfall Spike")
    st.line_chart(rain_df.set_index('time')['rain'])

    with st.expander("See Satellite Details"):
        st.write(f"Sentinel Status: {sentinel_status}")
        st.write(f"SMAP Status: {smap_status}")


if __name__ == "__main__":
    main()