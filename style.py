import streamlit as st


def apply_custom_css():
    st.markdown(
        """
        <style>
        /* 1. BACKGROUND UTAMA DASHBOARD */
        .stApp {
            background-color: #EEF2F6 !important;
            color: #1E293B !important;
        }

        /* 2. SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            padding-top: 1rem;
        }

        section[data-testid="stSidebar"] *, 
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] span {
            color: #475569 !important;
            font-weight: 500 !important;
        }

        section[data-testid="stSidebar"] a {
            border-radius: 8px !important;
            margin-bottom: 4px;
            transition: all 0.2s ease;
        }

        section[data-testid="stSidebar"] a:hover {
            background-color: #F1F5F9 !important;
            color: #2563EB !important;
        }

        section[data-testid="stSidebar"] a[aria-current="page"] {
            background-color: #EFF6FF !important;
            border-left: 4px solid #2563EB !important;
            border-radius: 4px 8px 8px 4px !important;
        }

        section[data-testid="stSidebar"] a[aria-current="page"] span,
        section[data-testid="stSidebar"] a[aria-current="page"] p {
            color: #2563EB !important;
            font-weight: 700 !important;
        }

        /* 3. PERBAIKAN TOTAL SEMUA TOMBOL (BUTTON, DOWNLOAD, & FORM SUBMIT) */
        div.stButton > button, 
        div.stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] > button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"],
        button[kind="primary"],
        button[kind="secondary"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
            transition: all 0.2s ease-in-out !important;
            width: 100% !important;
        }

        /* Memaksa Teks, Paragraph, dan Emoji di Dalam Tombol Berwarna Putih Tajam */
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

        /* Efek Hover Tombol */
        div.stButton > button:hover, 
        div.stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: #1D4ED8 !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 8px rgba(37, 99, 235, 0.35) !important;
            transform: translateY(-1px);
        }

        /* 4. METRIC CARD STYLING */
        .metric-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .metric-title {
            color: #64748B;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            color: #0F172A;
            font-size: 1.75rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .metric-subtitle {
            color: #2563EB;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* 5. SECTION HEADER STYLING */
        .section-header {
            color: #0F172A;
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid #CBD5E1;
        }

        /* 6. STYLING INPUT & WIDGET */
        div[data-testid="stTextInput"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stSlider"] label {
            color: #0F172A !important;
            font-weight: 700 !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
        }

        /* 7. CONTAINER & TABEL */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def render_header(title, subtitle):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1E3A8A 0%, #1E293B 100%);
            padding: 1.8rem 2.2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        ">
            <h1 style="color: #FFFFFF; font-size: 1.8rem; font-weight: 800; margin: 0 0 0.4rem 0; letter-spacing: -0.02em;">
                {title}
            </h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin: 0; font-weight: 400;">
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