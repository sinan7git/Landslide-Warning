import streamlit as st
import ee
import geemap.foliumap as geemap
import requests
import pandas as pd
import datetime

# --- CONFIGURATION ---
PROJECT_ID = ''  # Project ID


# --- GEOCODING FUNCTION (FREE - OpenStreetMap Nominatim) ---
def search_location(place_name):
    """Search for a place and get coordinates using free Nominatim API"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': place_name,
            'format': 'json',
            'limit': 5
        }
        headers = {
            'User-Agent': 'HillSafe-LandslideWarning/1.0'
        }

        response = requests.get(url, params=params, headers=headers)
        results = response.json()

        if results:
            return results
        else:
            return None
    except Exception as e:
        st.error(f"Cannot search location: {e}")
        return None


# --- 1. INITIALIZE SATELLITE CONNECTION ---
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize(project=PROJECT_ID)
        return True
    except Exception as e:
        try:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID)
            return True
        except Exception as auth_error:
            st.error(f"Cannot connect to satellite system: {auth_error}")
            return False


# --- 2. FETCH RAIN FORECAST (Next 2 Days) ---
def get_rainfall_data(lat, lon):
    try:
        # Request forecast for next 2 days (48 hours)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=rain,precipitation_probability&forecast_days=2"
        response = requests.get(url)
        data = response.json()

        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])

        # 1. Total Rain Expected (Next 48h)
        total_rain_forecast = df['rain'].sum()

        # 2. Max Probability (Highest chance of rain in the next 48h)
        max_chance = df['precipitation_probability'].max()

        return total_rain_forecast, max_chance, df
    except Exception as e:
        st.error(f"Cannot get forecast: {e}")
        return 0, 0, pd.DataFrame()


# --- 3. FETCH SENTINEL-1 (Scientific Anomaly Detection) ---
def get_sentinel_stability(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(500)

        # 1. Get the "Current" Image (Last 12 days)
        now = datetime.datetime.now()
        current_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                              .filterBounds(roi)
                              .filterDate(now - datetime.timedelta(days=12), now)
                              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                              .filter(ee.Filter.eq('instrumentMode', 'IW'))
                              .select('VV'))

        if current_collection.size().getInfo() == 0:
            return "No recent pass", 0.0, "N/A"

        current_img = current_collection.sort('system:time_start', False).first()
        current_date = ee.Date(current_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

        # 2. Get the "Baseline" (Average of the last 3 months, EXCLUDING current week)
        baseline_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                               .filterBounds(roi)
                               .filterDate(now - datetime.timedelta(days=90), now - datetime.timedelta(days=15))
                               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                               .filter(ee.Filter.eq('instrumentMode', 'IW'))
                               .select('VV'))

        if baseline_collection.size().getInfo() == 0:
            return "No baseline data", 0.0, "N/A"

        baseline_mean = baseline_collection.mean()

        # 3. Calculate the Anomaly (Difference from Normal)
        diff = current_img.subtract(baseline_mean).abs()

        # Reduce to a score
        change_score = diff.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        ).getInfo()['VV']

        return "Active", change_score, current_date

    except Exception as e:
        return f"Error: {e}", 0.0, "Error"


# --- 4. FETCH NASA SMAP (Soil Moisture Fallback) ---
def get_smap_moisture(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])

        # Look at the last 3 days for SMAP data
        now = datetime.datetime.now()
        start_date = (now - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')

        dataset = (ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006")
                   .filterBounds(point)
                   .filterDate(start_date, end_date)
                   .select('soil_moisture_am'))

        count = dataset.size().getInfo()

        if count == 0:
            return "No recent data", 0.0, "N/A"

        # Get the latest image date
        latest_img = dataset.sort('system:time_start', False).first()
        last_pass_date = ee.Date(latest_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

        # Get average moisture
        mean_img = dataset.mean()
        moisture_val = mean_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=9000
        ).getInfo()['soil_moisture_am']

        return "Working", moisture_val, last_pass_date

    except Exception as e:
        return f"Error: {e}", 0.0, "Error"


# --- 5. LOCATION SETUP UI ---
def setup_location():
    """Allow user to search and select location"""
    st.sidebar.title("📍 Set Your Location")

    # Initialize session state
    if 'location_name' not in st.session_state:
        st.session_state.location_name = "Chooralmala, Wayanad"
        st.session_state.lat = 11.54
        st.session_state.lon = 76.13

    # Search box
    search_query = st.sidebar.text_input(
        "Search for your village/town:",
        placeholder="e.g., Wayanad, Kerala"
    )

    if st.sidebar.button("🔍 Search Location"):
        if search_query:
            with st.spinner("Searching..."):
                results = search_location(search_query)

                if results:
                    st.session_state.search_results = results
                else:
                    st.sidebar.error("No results found. Try different spelling.")

    # Display search results
    if 'search_results' in st.session_state:
        st.sidebar.subheader("Select from results:")

        for i, result in enumerate(st.session_state.search_results):
            display_name = result.get('display_name', 'Unknown')
            lat = float(result['lat'])
            lon = float(result['lon'])

            if st.sidebar.button(f"📌 {display_name}", key=f"loc_{i}"):
                st.session_state.location_name = display_name
                st.session_state.lat = lat
                st.session_state.lon = lon
                st.session_state.pop('search_results')  # Clear results
                st.rerun()

    # Show current location
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Location:**")
    st.sidebar.info(f"📍 {st.session_state.location_name}")
    st.sidebar.caption(f"Lat: {st.session_state.lat:.4f}, Lon: {st.session_state.lon:.4f}")

    # Manual coordinate entry (advanced option)
    with st.sidebar.expander("⚙️ Enter Coordinates Manually"):
        manual_lat = st.number_input("Latitude", value=st.session_state.lat, format="%.6f")
        manual_lon = st.number_input("Longitude", value=st.session_state.lon, format="%.6f")
        manual_name = st.text_input("Location Name", value=st.session_state.location_name)

        if st.button("Set Manual Location"):
            st.session_state.lat = manual_lat
            st.session_state.lon = manual_lon
            st.session_state.location_name = manual_name
            st.rerun()


# --- 6. EDUCATION SECTION (Village-Friendly Explanation) ---
def show_education_section():
    with st.expander("ℹ️ How to read this? (Click for simple explanation)"):
        st.markdown("""
        ### Simple Guide for Everyone

        **1. 🌧️ What is 'Rain Prediction'?**
        * The satellite looks at the clouds coming tomorrow.
        * **Prediction:** We are showing you the rain expected for the **next 2 days**.
        * **Danger:** If it says >80mm is coming, prepare your drains.

        **2. 💧 What is 'Soil Wetness' (SMAP)?**
        * Think of the hill like a **Kitchen Sponge**.
        * **0.4 (40%)**: The sponge is getting full.
        * **Higher**: The hill is heavy with water and might slip.

        **3. 📡 What is 'Ground Movement' (Radar)?**
        * We compare the hill today vs. the last 3 months.
        * If this number is high (>2.0), the **texture** of the ground has changed (mud, sliding soil, or lost trees).
        """)

# --- 7. THE MAIN DASHBOARD ---
def main():
    st.set_page_config(page_title="Landslide Warning", page_icon="⛰️", layout="wide")

    # Location Setup in Sidebar
    setup_location()

    # Get current location
    LOCATION_NAME = st.session_state.location_name
    LAT = st.session_state.lat
    LON = st.session_state.lon

    st.title("⛰️ Landslide Warning System")
    st.markdown(f"**Your Area:** {LOCATION_NAME}")

    # --- INSERT THE HELP SECTION ---
    show_education_section()

    st.markdown("---")

    with st.spinner('Checking satellites...'):
        connected = initialize_earth_engine()

    if not connected:
        st.warning("Connecting to system...")
        return

    # --- EXECUTE DATA FETCHING ---
    rain_forecast, rain_chance, rain_df = get_rainfall_data(LAT, LON)
    sentinel_status, sentinel_score, sentinel_date = get_sentinel_stability(LAT, LON)
    smap_status, smap_score, smap_date = get_smap_moisture(LAT, LON)

    # --- HYBRID RISK ENGINE ---
    risk_factors = []

    # Factor 1: Rain Forecast
    if rain_forecast > 80:
        risk_factors.append(f"⚠️ Heavy Rain Coming ({rain_forecast:.1f}mm)")

    if rain_chance > 90:
        risk_factors.append("☁️ 90% Chance of Storm")

    # Factor 2: Structural Instability (Sentinel)
    if sentinel_score > 2.0:
        risk_factors.append("Hill Surface Changed (Radar)")

    # Factor 3: Soil Saturation (SMAP)
    if isinstance(smap_score, float) and smap_score > 0.4:
        risk_factors.append(f"Soil Too Wet ({(smap_score * 100):.1f}%)")

    # Determine Level
    risk_level = "SAFE"
    risk_color = "green"

    if len(risk_factors) == 1:
        risk_level = "BE CAREFUL"
        risk_color = "orange"
    elif len(risk_factors) >= 2:
        risk_level = "DANGER - LEAVE NOW"
        risk_color = "red"

    # --- UI DISPLAY ---

    # 1. Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌧️ Rain Forecast (48h)",
                f"{rain_forecast:.1f} mm" if rain_forecast is not None else "N/A",
                help="Total expected rain for the next 2 days")
    col2.metric("📡 Ground Movement",
                f"{sentinel_score:.2f}" if sentinel_score is not None else "N/A",
                help="Anomaly Score (Today vs 90-Day Avg)")
    col3.metric("💧 Soil Wetness",
                f"{smap_score:.3f}" if smap_score is not None else "N/A",
                help="Above 0.4 means too wet")
    col4.markdown(f"### Warning: :{risk_color}[{risk_level}]")

    if risk_factors:
        st.warning(f"⚠️ Danger Signs: {', '.join(risk_factors)}")

    # 2. Charts & Maps
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### 🌧️ Rainfall Prediction (Next 48 Hours)")
        if not rain_df.empty:
            st.line_chart(rain_df.set_index('time')['rain'])
    with c2:
        st.markdown("### 📍 Your Location")
        map_data = pd.DataFrame({'lat': [LAT], 'lon': [LON]})
        st.map(map_data, zoom=11)

    # 3. SOURCE INTELLIGENCE SECTION
    st.markdown("---")
    st.subheader("🛰️ Data Sources (Last Updated)")

    src1, src2, src3 = st.columns(3)

    with src1:
        st.info("**Satellite Radar**")
        st.write(f"**Status:** {sentinel_status}")
        st.write(f"**Last Check:** {sentinel_date}")
        if sentinel_date != "N/A" and sentinel_date != "Error":
            try:
                days_lag = (datetime.datetime.now() - pd.to_datetime(sentinel_date)).days
                if days_lag > 10:
                    st.caption("⚠️ Old data (using backup)")
                else:
                    st.caption("✅ Recent data")
            except:
                pass

    with src2:
        st.info("**Soil Wetness Monitor**")
        st.write(f"**Status:** {smap_status}")
        st.write(f"**Last Check:** {smap_date}")
        st.caption("Backup data source")

    with src3:
        st.info("**Weather Forecast**")
        st.write("**Status:** Working")
        st.write("**Prediction:** Next 48 Hours")
        st.caption("Main warning system")


if __name__ == "__main__":
    main()