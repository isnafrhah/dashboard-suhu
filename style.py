import streamlit as st


def apply_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

        /* 1. BACKGROUND & TEMA UTAMA DASHBOARD (FOREST GREEN + SAGE + WARM CREAM) */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            overflow-x: hidden !important;
            max-width: 100vw !important;
            background-color: #F8F6EF !important;
            color: #29352F !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1.8rem !important;
            padding-bottom: 3.5rem;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Manrope', sans-serif !important;
            color: #174C3A !important;
            letter-spacing: -0.02em;
        }

        /* 2. SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: #174C3A !important;
            border-right: none !important;
            padding-top: 1rem;
        }

        section[data-testid="stSidebar"] *, 
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] span {
            color: #E2ECE5 !important;
            font-weight: 500 !important;
        }

        section[data-testid="stSidebar"] a {
            border-radius: 8px !important;
            margin-bottom: 4px;
            transition: all 0.2s ease;
        }

        section[data-testid="stSidebar"] a:hover {
            background-color: rgba(255, 255, 255, 0.12) !important;
            color: #FFFFFF !important;
        }

        section[data-testid="stSidebar"] a[aria-current="page"] {
            background-color: #D6AD55 !important;
            border-left: 4px solid #B8892E !important;
            border-radius: 4px 8px 8px 4px !important;
        }

        section[data-testid="stSidebar"] a[aria-current="page"] span,
        section[data-testid="stSidebar"] a[aria-current="page"] p {
            color: #174C3A !important;
            font-weight: 800 !important;
        }

        /* 3. TOMBOL UTAMA (FOREST GREEN #174C3A) */
        div.stButton > button, 
        div.stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] > button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"],
        button[kind="primary"],
        button[kind="secondary"] {
            background-color: #174C3A !important;
            color: #FFFFFF !important;
            border: 1px solid #10382B !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.4rem !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            box-shadow: 0 3px 10px rgba(23, 76, 58, 0.22) !important;
            transition: all 0.2s ease-in-out !important;
            width: 100% !important;
        }

        div.stButton > button *, 
        div.stDownloadButton > button *,
        div[data-testid="stFormSubmitButton"] > button *,
        div[data-testid="stFormSubmitButton"] button p,
        button[data-testid="baseButton-secondary"] *,
        button[data-testid="baseButton-primary"] * {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }

        div.stButton > button:hover, 
        div.stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: #10382B !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 16px rgba(23, 76, 58, 0.35) !important;
            transform: translateY(-1px);
        }

        /* 4. METRIC CARD STYLING (SAGE GREEN + FOREST GREEN ACCENT) */
        .metric-card {
            background-color: #F0F6F2;
            border: 1px solid #A8C3AD;
            border-top: 3.5px solid #174C3A;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 2px 8px rgba(23, 76, 58, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .metric-card:hover {
            background-color: #FFFFFF;
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(23, 76, 58, 0.1);
        }
        .metric-title {
            color: #54665C;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            color: #174C3A;
            font-family: 'Manrope', sans-serif;
            font-size: 1.6rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            line-height: 1.2;
        }
        .metric-subtitle {
            color: #2D725A;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* 5. SECTION HEADER STYLING */
        .section-header {
            color: #174C3A;
            font-family: 'Manrope', sans-serif;
            font-size: 1.15rem;
            font-weight: 800;
            margin-top: 1.6rem;
            margin-bottom: 1rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid #A8C3AD;
        }

        /* 6. STYLING INPUT & WIDGET */
        div[data-testid="stTextInput"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stDateInput"] label {
            color: #174C3A !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input {
            background-color: #FFFFFF !important;
            color: #29352F !important;
            border: 1.5px solid #A8C3AD !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stDateInput"] input:focus {
            border-color: #174C3A !important;
            box-shadow: 0 0 0 3px rgba(23, 76, 58, 0.18) !important;
        }

        /* 7. CONTAINER & TABEL */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background-color: #FFFFFF !important;
            border: 1.5px solid #A8C3AD !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(23, 76, 58, 0.05);
            max-width: 100% !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
        }

        /* 8. GRAFIK PLOTLY */
        div[data-testid="stPlotlyChart"],
        .js-plotly-plot, .plot-container {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        /* 9. RESPONSIVITAS MULTI-SCREEN */
        @media (max-width: 992px) {
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 12px !important;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 3.2rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-bottom: 2rem !important;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                width: 100% !important;
            }
            .metric-card {
                padding: 12px 14px;
                border-radius: 12px;
                margin-bottom: 8px;
            }
            .metric-value {
                font-size: 1.35rem;
            }
            div.stButton > button, 
            div.stDownloadButton > button,
            div[data-testid="stFormSubmitButton"] > button {
                padding: 0.75rem 1rem !important;
                font-size: 0.92rem !important;
                min-height: 44px !important;
            }
            div[data-testid="stTextInput"] input,
            div[data-testid="stNumberInput"] input,
            div[data-testid="stDateInput"] input {
                font-size: 16px !important; /* Mencegah auto zoom iOS */
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def render_header(title, subtitle):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(105deg, #1F5C48 0%, #174C3A 55%, #10382B 100%);
            padding: 1.6rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 6px 20px rgba(23, 76, 58, 0.2);
            border: 1px solid rgba(255,255,255,0.12);
        ">
            <div style="font-size: 0.68rem; font-weight: 800; letter-spacing: 0.12em; color: #D6AD55; text-transform: uppercase; margin-bottom: 6px;">
                CUACATANI CABAI · SISTEM INFORMASI CUACA UNTUK PETANI CABAI
            </div>
            <h1 style="color: #FFFFFF; font-family: 'Manrope', sans-serif; font-size: clamp(1.25rem, 2.5vw, 1.7rem); font-weight: 800; margin: 0 0 0.4rem 0; letter-spacing: -0.02em; line-height: 1.25;">
                {title}
            </h1>
            <p style="color: rgba(255,255,255,0.85); font-size: 0.88rem; margin: 0; font-weight: 400; line-height: 1.5;">
                {subtitle}
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def draw_metric_card(title, value, subtitle):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )
