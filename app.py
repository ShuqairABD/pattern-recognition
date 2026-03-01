# worked good 😏
"""
app.py — Pattern Recognition System
Run: python app.py → http://127.0.0.1:8050
"""

import ssl, os, certifi
os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from pattern_recognition import PatternSystem
import requests, base64, io, json

INSTRUMENT_GROUPS = {
    "Crypto": {
        "BTC":  "BTC-USD",
        "ETH":  "ETH-USD",
        "SOL":  "SOL-USD",
        "BNB":  "BNB-USD",
        "XRP":  "XRP-USD",
        "ADA":  "ADA-USD",
        "DOGE": "DOGE-USD",
    },
    "Stocks": {
        "S&P500": "SPY",
        "Nasdaq": "QQQ",
        "Apple":  "AAPL",
        "NVIDIA": "NVDA",
        "Tesla":  "TSLA",
        "Amazon": "AMZN",
        "MSFT":   "MSFT",
        "Google": "GOOGL",
    },
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CHF": "CHF=X",
    },
    "Commodities": {
        "Gold":   "GC=F",
        "Silver": "SI=F",
        "Oil":    "BZ=F",
        "Copper": "HG=F",
        "Gas":    "NG=F",
        "Wheat":  "ZW=F",
    },
}

ALL_INSTRUMENTS = {}
for _g in INSTRUMENT_GROUPS.values():
    ALL_INSTRUMENTS.update(_g)

# Big TF options: label → (interval, period)
TF_BIG_OPTIONS = {
    "1 Day":    ("1d",  "2y"),
    "4 Hours":  ("4h",  "60d"),
    "1 Hour":   ("1h",  "30d"),
    "15 Min":   ("15m", "8d"),
    "5 Min":    ("5m",  "5d"),
}

# Small TF options: label → (interval, period)
# Periods chosen to give ~300-500 bars for pattern detection
TF_SMALL_OPTIONS = {
    "4 Hours":  ("4h",  "90d"),
    "1 Hour":   ("1h",  "30d"),
    "15 Min":   ("15m", "14d"),
    "5 Min":    ("5m",  "5d"),
    "1 Min":    ("1m",  "2d"),
}

# Default small TF for each big TF (auto-suggest, user can override)
TF_DEFAULT_SMALL = {
    "1 Day":   "4 Hours",
    "4 Hours": "1 Hour",
    "1 Hour":  "15 Min",
    "15 Min":  "5 Min",
    "5 Min":   "1 Min",
}

# Keep for backward compat in places that need (tf_big, period_big, tf_small, period_small)
TF_OPTIONS = {
    "1 Day":   ("1d",  "2y",  "1h",  "60d"),
    "4 Hours": ("4h",  "60d", "1h",  "14d"),
    "1 Hour":  ("1h",  "30d", "15m", "5d"),
    "15 Min":  ("15m", "8d",  "5m",  "2d"),
    "5 Min":   ("5m",  "5d",  "1m",  "1d"),
}

TF_PARAMS = {
    "1d":  dict(smooth_window=13, min_ext_dist=5, lambda1=0.80, lambda2=0.65, quality_thresh=0.50),
    "4h":  dict(smooth_window=12, min_ext_dist=4, lambda1=0.78, lambda2=0.62, quality_thresh=0.47),
    "1h":  dict(smooth_window=11, min_ext_dist=4, lambda1=0.75, lambda2=0.60, quality_thresh=0.45),
    "15m": dict(smooth_window=9,  min_ext_dist=3, lambda1=0.70, lambda2=0.55, quality_thresh=0.42),
    "5m":  dict(smooth_window=7,  min_ext_dist=3, lambda1=0.65, lambda2=0.50, quality_thresh=0.40),
    "1m":  dict(smooth_window=5,  min_ext_dist=2, lambda1=0.60, lambda2=0.45, quality_thresh=0.38),
}

# Minutes per bar for each TF — used for time-correct box sizing
TF_MINUTES = {
    "1m":   1,
    "5m":   5,
    "15m":  15,
    "4h":   240,
    "1h":   60,
    "1d":   1440,
}

PHASE_COLORS = {1:"#00B4D8", 2:"#FFB703", 3:"#06D6A0", 0:"#888"}
IMP_COLOR    = "#06D6A0"
CORR_COLOR   = "#FFB703"
ZOOM_COLOR   = "#FF4444"

REFRESH_OPTIONS = {
    "Off":    None,
    "10 sec": 10_000,
    "30 sec": 30_000,
    "1 min":  60_000,
    "5 min":  300_000,
}

# Will be loaded dynamically from OpenRouter
AI_MODELS_DEFAULT = [
    {"label": "-- Press 'Load Models' first --", "value": "none"},
]

# Variant A: Preset analysis modes
AI_PRESET_MODES = [
    {"label": "🎯 Find Entry Point",     "value": "entry"},
    {"label": "📊 Market Overview",      "value": "overview"},
    {"label": "⚠️ Risk Assessment",      "value": "risk"},
    {"label": "🔄 Correction Analysis",  "value": "correction"},
    {"label": "✍️ Custom Question",      "value": "custom"},
]

# Variant B: System prompt presets (trading style)
AI_SYSTEM_PRESETS = [
    {"label": "🎯 Trend Follower",       "value": "trend"},
    {"label": "⚡ Scalper",              "value": "scalper"},
    {"label": "🛡️ Risk Manager",        "value": "risk_mgr"},
    {"label": "📐 Technical Analyst",    "value": "technical"},
]

AI_SYSTEM_PROMPTS = {
    "trend": (
        "You are a trend-following trader. "
        "ONLY recommend trades in the direction of the main trend. "
        "Always provide: entry price, stop-loss, take-profit (minimum 2:1 reward/risk). "
        "Risk per trade: maximum 2% of deposit. "
        "If trend is unclear - say WAIT, do not force a trade."
    ),
    "scalper": (
        "You are a scalper focused on short-term moves. "
        "Look for quick entries with tight stop-loss (0.5-1%). "
        "Targets: small but realistic (0.5-2%). "
        "Always specify exact entry, stop-loss, and take-profit levels. "
        "React to small TF patterns primarily."
    ),
    "risk_mgr": (
        "You are a conservative risk manager. "
        "Your priority is capital preservation. "
        "Maximum risk per trade: 1% of deposit. "
        "Only recommend HIGH confidence setups (3+ confirmations). "
        "Always warn about potential dangers before giving entry signals. "
        "If risk/reward is below 2:1 - do not recommend entry."
    ),
    "technical": (
        "You are a pure technical analyst. "
        "Base all analysis on chart patterns, wave structure (W1/W2/W3), "
        "support/resistance levels, and fractality. "
        "Always specify exact price levels. "
        "Provide probability estimate for each scenario (e.g. 65% bullish / 35% bearish)."
    ),
}

AI_SYSTEM_PROMPTS_RU = {
    "trend": (
        "Ты трейдер следующий за трендом. "
        "ТОЛЬКО рекомендуй сделки в направлении основного тренда. "
        "Всегда указывай: цена входа, стоп-лосс, тейк-профит (минимум 2:1). "
        "Риск на сделку: максимум 2% депозита. "
        "Если тренд не ясен — говори ЖДАТЬ, не форсируй сделку."
    ),
    "scalper": (
        "Ты скальпер, ориентированный на краткосрочные движения. "
        "Ищи быстрые входы с коротким стопом (0.5-1%). "
        "Цели: небольшие но реальные (0.5-2%). "
        "Всегда указывай точный вход, стоп-лосс и тейк-профит. "
        "Ориентируйся прежде всего на паттерны малого ТФ."
    ),
    "risk_mgr": (
        "Ты консервативный риск-менеджер. "
        "Приоритет — сохранение капитала. "
        "Максимальный риск на сделку: 1% депозита. "
        "Рекомендуй только ВЫСОКОКОНФИДЕНТНЫЕ сетапы (3+ подтверждения). "
        "Всегда предупреждай об опасностях перед сигналом на вход. "
        "Если соотношение риск/доходность ниже 2:1 — не рекомендуй вход."
    ),
    "technical": (
        "Ты чистый технический аналитик. "
        "Базируй анализ на паттернах, волновой структуре (W1/W2/W3), "
        "уровнях поддержки/сопротивления и фрактальности. "
        "Всегда указывай точные ценовые уровни. "
        "Давай оценку вероятности каждого сценария (например 65% бычий / 35% медвежий)."
    ),
}

AI_SYSTEM_PROMPTS_AR = {
    "trend": (
        "أنت متداول يتبع الاتجاه. "
        "أوصِ فقط بالصفقات في اتجاه الترند الرئيسي. "
        "أعطِ دائماً: سعر الدخول، وقف الخسارة، جني الأرباح (نسبة 2:1 على الأقل). "
        "المخاطرة لكل صفقة: 2% من رأس المال كحد أقصى. "
        "إذا كان الاتجاه غير واضح — قل انتظر، لا تدخل قسراً."
    ),
    "scalper": (
        "أنت متداول سكالبر تركز على تحركات قصيرة المدى. "
        "ابحث عن نقاط دخول سريعة مع وقف خسارة ضيق (0.5-1%). "
        "الأهداف: صغيرة لكن واقعية (0.5-2%). "
        "حدد دائماً سعر الدخول ووقف الخسارة وجني الأرباح. "
        "اعتمد أساساً على أنماط الإطار الزمني الصغير."
    ),
    "risk_mgr": (
        "أنت مدير مخاطر محافظ. "
        "أولويتك حماية رأس المال. "
        "المخاطرة القصوى لكل صفقة: 1% من رأس المال. "
        "أوصِ فقط بالإعدادات عالية الثقة (3+ تأكيدات). "
        "حذّر دائماً من المخاطر قبل إعطاء إشارة الدخول. "
        "إذا كانت نسبة المكافأة/المخاطرة أقل من 2:1 — لا توصِ بالدخول."
    ),
    "technical": (
        "أنت محلل تقني بحت. "
        "أسّس تحليلك على الأنماط وهيكل الأمواج (W1/W2/W3) "
        "ومستويات الدعم والمقاومة والكسورية. "
        "حدد دائماً مستويات الأسعار بدقة. "
        "أعطِ تقدير الاحتمالية لكل سيناريو (مثلاً 65% صاعد / 35% هابط)."
    ),
}

# ── Page translations ─────────────────────────────────────
TR = {
    "en": {
        "title":        "📊 Pattern Recognition System",
        "subtitle":     "Part resembles the whole",
        "instrument":   "Instrument",
        "timeframe":    "Timeframe",
        "box_mode":     "Box Mode",
        "by_time":      "⏱ By Time",
        "by_pattern":   "📐 By Pattern",
        "auto_refresh": "Auto-Refresh",
        "run":          "Run",
        "btn_analyse":  "🔄 Analyse",
        "btn_update":   "⚡ Update Now",
        "legend_imp":   " Impulse (W1,W3)  ",
        "legend_corr":  " Correction (W2)  ",
        "legend_here":  " YOU ARE HERE  ",
        "legend_mode":  "⏱ by time  |  📐 by pattern",
        "report_btn":   "📋 Report",
        "report_title": "Analysis Report",
        "ai_title":     "🤖 AI Analysis  ",
        "ai_sub":       "Vision + Data → OpenRouter",
        "ai_key_label": "OpenRouter API Key",
        "ai_load_btn":  "🔍 Load Models",
        "ai_model_lbl": "Model",
        "ai_lang_lbl":  "Response Language",
        "ai_btn":       "🧠 Analyse with AI",
        "ai_hint":      "Enter API key → Load Models → Run Analyse → Analyse with AI",
        "alert_init":   "Press Analyse to detect position",
        "dir":          "ltr",
    },
    "ar": {
        "title":        "📊 نظام التعرف على الأنماط",
        "subtitle":     "الجزء يشبه الكل",
        "instrument":   "الأداة",
        "timeframe":    "الإطار الزمني",
        "box_mode":     "وضع الإطار",
        "by_time":      "⏱ حسب الوقت",
        "by_pattern":   "📐 حسب النمط",
        "auto_refresh": "التحديث التلقائي",
        "run":          "تشغيل",
        "btn_analyse":  "🔄 تحليل",
        "btn_update":   "⚡ تحديث الآن",
        "legend_imp":   " اندفاع (W1,W3)  ",
        "legend_corr":  " تصحيح (W2)  ",
        "legend_here":  " أنت هنا  ",
        "legend_mode":  "⏱ حسب الوقت  |  📐 حسب النمط",
        "report_btn":   "📋 التقرير",
        "report_title": "تقرير التحليل",
        "ai_title":     "🤖 تحليل الذكاء الاصطناعي  ",
        "ai_sub":       "رؤية + بيانات → OpenRouter",
        "ai_key_label": "مفتاح OpenRouter API",
        "ai_load_btn":  "🔍 تحميل النماذج",
        "ai_model_lbl": "النموذج",
        "ai_lang_lbl":  "لغة الرد",
        "ai_btn":       "🧠 تحليل بالذكاء الاصطناعي",
        "ai_hint":      "أدخل المفتاح ← حمّل النماذج ← شغّل التحليل ← حلل بالذكاء الاصطناعي",
        "alert_init":   "اضغط تحليل لتحديد الموقع",
        "dir":          "rtl",
    },
}

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Pattern Recognition"
server = app.server  # needed for gunicorn / Render

GROUP_ICONS = {"Crypto":"🪙","Stocks":"📈","Forex":"💱","Commodities":"🛢️"}

def make_instrument_panel():
    cols = []
    for group_name, instruments in INSTRUMENT_GROUPS.items():
        icon = GROUP_ICONS.get(group_name,"")
        btns = html.Div([
            html.Small(f"{icon} {group_name}",
                className="text-warning d-block mb-1",
                style={"fontSize":"11px","fontWeight":"bold"}),
            html.Div([
                dbc.Button(name, id=f"btn-{sym}", n_clicks=0,
                    color="secondary", size="sm",
                    style={"marginRight":"3px","marginBottom":"3px",
                           "fontSize":"11px","padding":"2px 7px"})
                for name, sym in instruments.items()
            ], className="d-flex flex-wrap"),
        ], className="me-3 mb-2")
        cols.append(btns)
    return html.Div(cols, className="d-flex flex-wrap")

app.layout = dbc.Container([
    dcc.Interval(id="auto-refresh", interval=999_999_999, disabled=True, n_intervals=0),
    dcc.Store(id="selected-symbol", data="GC=F"),
    dcc.Store(id="page-lang", data="en"),

    dbc.Row([
        dbc.Col(html.H3(id="ui-title", children="📊 Pattern Recognition System",
                        className="text-light my-2"), width=8),
        dbc.Col([
                # Language buttons hidden for now
                html.Div(id="btn-lang-en", style={"display":"none"}),
                html.Div(id="btn-lang-ar", style={"display":"none"}),
            ], width=2),
        dbc.Col(html.P(id="ui-subtitle", children="Part resembles the whole",
                       className="text-secondary my-2 text-end fst-italic"), width=2),
    ]),

    dbc.Card([dbc.CardBody([
        # Row 1: Instruments
        dbc.Row([
            dbc.Col([
                html.Label(id="lbl-instrument", children="Instrument", className="text-warning fw-bold mb-1"),
                make_instrument_panel(),
            ], width=12),
        ], className="mb-2"),
        # Row 2: Settings
        dbc.Row([
            dbc.Col([
                html.Label(id="lbl-timeframe", children="Big TF (Mother)", className="text-warning fw-bold mb-1"),
                dbc.RadioItems(id="tf-selector",
                    options=[{"label": k, "value": k} for k in TF_BIG_OPTIONS],
                    value="1 Day", inline=True, className="text-light small"),
                html.Div(style={"height":"6px"}),
                html.Label(id="lbl-tf-small", children="Small TF (Inside)", className="text-info fw-bold mb-1",
                           style={"fontSize":"12px"}),
                dbc.RadioItems(id="tf-small-selector",
                    options=[{"label": k, "value": k} for k in TF_SMALL_OPTIONS],
                    value="1 Hour", inline=True, className="text-light small"),
            ], width=5),
            dbc.Col([
                # Box Mode — always "By Pattern" (By Time removed)
                html.Div(id="lbl-boxmode", style={"display":"none"}),
                dcc.Store(id="box-mode", data="pattern"),
            ], width=3),
            dbc.Col([
                html.Label(id="lbl-refresh", children="Auto-Refresh", className="text-warning fw-bold mb-1"),
                dbc.Select(id="refresh-select",
                    options=[{"label": k, "value": k} for k in REFRESH_OPTIONS],
                    value="Off",
                    style={"backgroundColor":"#2a2a3e","color":"#fff",
                           "border":"1px solid #555","fontSize":"12px",
                           "height":"32px","padding":"4px 8px"}),
                html.Div(id="next-refresh-text", className="text-secondary mt-1",
                         style={"fontSize":"11px"}),
            ], width=2),
            dbc.Col([
                dbc.Button("🔄 Analyse", id="btn-run",
                           color="success", size="md", className="w-100 mb-1"),
                dbc.Button("⚡ Update Now", id="btn-refresh-now",
                           color="warning", size="sm", className="w-100"),
            ], width=2),
            dbc.Col([
                html.Div(id="status-text", className="text-info small"),
                html.Div(id="phase-badge", className="mt-1"),
            ], width=1),
        ], align="end"),
    ])], className="mb-2 bg-dark border-secondary"),

    dbc.Alert(id="position-alert", children="Press Analyse to detect position",
              color="dark", className="py-2 mb-2 border-secondary text-center fw-bold"),

    dbc.Alert([
        html.Span("🟩", style={"color":IMP_COLOR}),
        html.Span(id="leg-imp", children=" Impulse (W1,W3)  ", className="text-light me-3"),
        html.Span("🟨", style={"color":CORR_COLOR}),
        html.Span(id="leg-corr", children=" Correction (W2)  ", className="text-light me-3"),
        html.Span("🟥", style={"color":ZOOM_COLOR}),
        html.Span(id="leg-here", children=" YOU ARE HERE  ", className="text-light me-3"),
        html.Span(id="leg-mode", children="⏱ by time  |  📐 by pattern", className="text-secondary"),
    ], color="dark", className="py-1 mb-2 border-secondary small"),

    dbc.Spinner(
        dcc.Graph(id="main-chart", style={"height":"960px"}, config={"scrollZoom":True}),
        color="success", type="grow"
    ),

    dbc.Row([
        dbc.Col(dbc.Button(id="toggle-report", children="📋 Report",
            color="outline-secondary", size="sm"), width="auto"),
    ], className="mt-2"),

    dbc.Collapse(
        dbc.Card([
            dbc.CardHeader(id="report-header", children="Analysis Report", className="text-warning bg-dark"),
            dbc.CardBody(html.Pre(id="report-text", className="text-light small",
                style={"maxHeight":"280px","overflowY":"auto",
                       "fontFamily":"Segoe UI, Tahoma, Arial, monospace",
                       "fontSize":"12px"}))
        ], className="bg-dark border-secondary mt-1"),
        id="report-collapse", is_open=False
    ),

    # ── AI Analysis Panel ──────────────────────────────────
    dbc.Card([
        dbc.CardHeader([
            html.Span(id="ai-panel-title", children="🤖 AI Analysis  ", className="text-warning fw-bold"),
            html.Small(id="ai-panel-sub", children="Vision + Data → OpenRouter", className="text-secondary"),
        ], className="bg-dark py-2"),
        dbc.CardBody([
            # Row 1: Key + Model + Language
            dbc.Row([
                dbc.Col([
                    html.Label(id="lbl-apikey", children="OpenRouter API Key", className="text-warning small fw-bold mb-1"),
                    dbc.InputGroup([
                        dbc.Input(id="ai-api-key", type="password",
                            placeholder="sk-or-v1-...",
                            style={"backgroundColor":"#2a2a3e","color":"#fff",
                                   "border":"1px solid #555","fontSize":"13px"}),
                        dbc.Button(id="btn-load-models", children="🔍 Load Models",
                            color="info", size="sm", style={"whiteSpace":"nowrap"}),
                    ]),
                    html.Div(id="models-status", className="text-secondary mt-1",
                             style={"fontSize":"11px"}),
                ], width=4),
                dbc.Col([
                    html.Label(id="lbl-aimodel", children="Model", className="text-warning small fw-bold mb-1"),
                    dbc.Select(id="ai-model",
                        options=AI_MODELS_DEFAULT,
                        value="none",
                        style={"backgroundColor":"#2a2a3e","color":"#fff",
                               "border":"1px solid #555","fontSize":"12px"}),
                ], width=4),
                dbc.Col([
                    html.Label(id="lbl-ailang", children="Response Language", className="text-warning small fw-bold mb-1"),
                    dbc.RadioItems(id="ai-lang",
                        options=[{"label":"🇷🇺 Русский","value":"ru"},
                                 {"label":"🇬🇧 English","value":"en"},
                                 {"label":"🇸🇦 عربي","value":"ar"}],
                        value="ru", inline=True, className="text-light small"),
                ], width=4),
            ], className="mb-2 align-items-end"),

            # Row 2: Variant A — Preset mode + Variant B — System prompt + Run button
            dbc.Row([
                dbc.Col([
                    html.Label("🎯 Analysis Mode", className="text-success small fw-bold mb-1"),
                    dbc.Select(id="ai-preset-mode",
                        options=AI_PRESET_MODES,
                        value="overview",
                        style={"backgroundColor":"#1a2e1a","color":"#06D6A0",
                               "border":"1px solid #06D6A0","fontSize":"12px"}),
                ], width=4),
                dbc.Col([
                    html.Label("🧠 Trading Style (System Prompt)", className="text-warning small fw-bold mb-1"),
                    dbc.Select(id="ai-system-preset",
                        options=AI_SYSTEM_PRESETS,
                        value="trend",
                        style={"backgroundColor":"#2a2a1a","color":"#FFB703",
                               "border":"1px solid #FFB703","fontSize":"12px"}),
                ], width=4),
                dbc.Col([
                    html.Br(),
                    dbc.Button(id="btn-ai-analyse", children="🧠 Analyse with AI",
                        color="primary", size="md", className="w-100",
                        style={"background":"linear-gradient(135deg,#4361EE,#7B2FBE)",
                               "border":"none","fontWeight":"bold"}),
                ], width=4),
            ], className="mb-2 align-items-end"),

            # Row 3: Custom question (shown only when mode=custom)
            dbc.Collapse(
                dbc.Row([
                    dbc.Col([
                        dbc.Textarea(id="ai-custom-question",
                            placeholder="Type your question about the chart...",
                            style={"backgroundColor":"#2a2a3e","color":"#fff",
                                   "border":"1px solid #7B2FBE","fontSize":"13px",
                                   "resize":"vertical"},
                            rows=2),
                    ], width=12),
                ], className="mb-2"),
                id="custom-question-collapse", is_open=False
            ),

            # Response area
            dbc.Spinner([
                dbc.Alert(id="ai-response",
                    children=html.Div(
                        html.Span(id="ai-hint-text", children="Enter API key → Load Models → Run Analyse → Analyse with AI"),
                        className="text-secondary text-center py-3"),
                    color="dark", className="mb-0 border-secondary",
                    style={"minHeight":"100px","maxHeight":"500px",
                           "overflowY":"auto","whiteSpace":"pre-wrap",
                           "fontSize":"14px","lineHeight":"1.7"}),
            ], color="primary", type="border"),
        ], className="bg-dark"),
    ], className="mt-3 border-secondary"),

], fluid=True, id="main-container", className="bg-dark min-vh-100 px-4 pb-4")


# ══════════════════════════════════════════════════════════
# PAGE LANGUAGE CALLBACK
# ══════════════════════════════════════════════════════════
@app.callback(
    Output("page-lang",        "data"),
    Output("main-container",   "style"),
    Output("ui-title",         "children"),
    Output("ui-subtitle",      "children"),
    Output("lbl-instrument",   "children"),
    Output("lbl-timeframe",    "children"),
    Output("lbl-boxmode",      "children"),
    Output("lbl-refresh",      "children"),
    Output("btn-run",          "children"),
    Output("btn-refresh-now",  "children"),
    Output("leg-imp",          "children"),
    Output("leg-corr",         "children"),
    Output("leg-here",         "children"),
    Output("leg-mode",         "children"),
    Output("toggle-report",    "children"),
    Output("report-header",    "children"),
    Output("ai-panel-title",   "children"),
    Output("ai-panel-sub",     "children"),
    Output("lbl-apikey",       "children"),
    Output("btn-load-models",  "children"),
    Output("lbl-aimodel",      "children"),
    Output("lbl-ailang",       "children"),
    Output("btn-ai-analyse",   "children"),
    Output("ai-hint-text",     "children"),
    Output("report-text",      "style"),
    Input("btn-lang-en",       "n_clicks"),
    Input("btn-lang-ar",       "n_clicks"),
    State("page-lang",         "data"),
    prevent_initial_call=True
)
def switch_language(n_en, n_ar, current_lang):
    ctx = callback_context
    if not ctx.triggered:
        lang = current_lang or "en"
    else:
        btn = ctx.triggered[0]["prop_id"]
        lang = "ar" if "btn-lang-ar" in btn else "en"

    t = TR[lang]
    container_style = {
        "direction": t["dir"],
        "textAlign": "right" if t["dir"] == "rtl" else "left",
    }
    return (
        lang,
        container_style,
        t["title"], t["subtitle"],
        t["instrument"], t["timeframe"],
        t["box_mode"], t["auto_refresh"],
        t["btn_analyse"], t["btn_update"],
        t["legend_imp"], t["legend_corr"],
        t["legend_here"], t["legend_mode"],
        t["report_btn"], t["report_title"],
        t["ai_title"], t["ai_sub"],
        t["ai_key_label"], t["ai_load_btn"],
        t["ai_model_lbl"], t["ai_lang_lbl"],
        t["ai_btn"], t["ai_hint"],
        {"maxHeight":"280px","overflowY":"auto",
         "fontFamily":"Segoe UI, Tahoma, Arial, monospace",
         "fontSize":"12px",
         "direction": t["dir"],
         "textAlign": "right" if t["dir"]=="rtl" else "left"},
    )


@app.callback(
    Output("auto-refresh","interval"),
    Output("auto-refresh","disabled"),
    Output("next-refresh-text","children"),
    Input("refresh-select","value"),
    prevent_initial_call=True,
)
def set_refresh(choice):
    ms = REFRESH_OPTIONS.get(choice)
    if ms is None:
        return 999_999_999, True, "Auto-refresh off"
    return ms, False, f"⟳ every {ms//1000} sec"


# ── Auto-suggest small TF when big TF changes ─────────────
@app.callback(
    Output("tf-small-selector", "value"),
    Input("tf-selector", "value"),
    State("tf-small-selector", "value"),
    prevent_initial_call=True
)
def suggest_small_tf(big_tf_name, current_small):
    suggested = TF_DEFAULT_SMALL.get(big_tf_name, "1 Hour")
    # Only change if current small TF is not compatible
    big_interval = TF_BIG_OPTIONS.get(big_tf_name, ("1d",""))[0]
    small_interval = TF_SMALL_OPTIONS.get(current_small, ("1h",""))[0]
    # Prevent small >= big
    order = ["1d","4h","1h","15m","5m","1m"]
    big_idx   = order.index(big_interval)   if big_interval   in order else 0
    small_idx = order.index(small_interval) if small_interval in order else 2
    if small_idx <= big_idx:
        return suggested
    return current_small


@app.callback(
    Output("selected-symbol","data"),
    Output("status-text","children"),
    [Input(f"btn-{sym}","n_clicks") for sym in ALL_INSTRUMENTS.values()],
    State("selected-symbol","data"),
    prevent_initial_call=True
)
def select_instrument(*args):
    ctx = callback_context
    if not ctx.triggered:
        return args[-1], ""
    sym  = ctx.triggered[0]["prop_id"].split(".")[0].replace("btn-","")
    name = next((n for n,s in ALL_INSTRUMENTS.items() if s==sym), sym)
    return sym, f"Selected: {name} ({sym})"


@app.callback(
    Output("main-chart","figure"),
    Output("report-text","children"),
    Output("phase-badge","children"),
    Output("position-alert","children"),
    Output("position-alert","color"),
    Output("report-collapse","is_open"),
    Input("btn-run","n_clicks"),
    Input("btn-refresh-now","n_clicks"),
    Input("auto-refresh","n_intervals"),
    State("selected-symbol","data"),
    State("tf-selector","value"),
    State("tf-small-selector","value"),
    State("box-mode","value"),
    State("page-lang","data"),
    prevent_initial_call=True
)
def run_analysis(n1, n2, n_auto, symbol, tf_name, tf_small_name, box_mode, page_lang):
    page_lang  = page_lang or "en"
    tf_big,    period_big   = TF_BIG_OPTIONS.get(tf_name,   ("1d", "2y"))
    tf_small,  period_small = TF_SMALL_OPTIONS.get(tf_small_name, ("1h", "14d"))
    try:
        pb = _load(symbol, period_big,   tf_big)
        ps = _load(symbol, period_small, tf_small)
    except Exception as e:
        return _error_fig(str(e)), str(e), "", f"Error: {e}", "danger", True

    p_big   = TF_PARAMS.get(tf_big,   TF_PARAMS["1d"])
    p_small = TF_PARAMS.get(tf_small, TF_PARAMS["1h"])

    sb = PatternSystem(smooth_window=p_big["smooth_window"],   smooth_poly=3,
        min_ext_dist=p_big["min_ext_dist"],
        lambda1=p_big["lambda1"],   lambda2=p_big["lambda2"],
        alpha=0.618, r_min=0.25, r_max=0.85, quality_thresh=p_big["quality_thresh"])
    ss = PatternSystem(smooth_window=p_small["smooth_window"], smooth_poly=3,
        min_ext_dist=p_small["min_ext_dist"],
        lambda1=p_small["lambda1"], lambda2=p_small["lambda2"],
        alpha=0.50,  r_min=0.20, r_max=0.90, quality_thresh=p_small["quality_thresh"])

    rb = sb.run(pb)
    rs = ss.run(ps)
    frac = sb.fractal.self_similarity(rs["waves"], rb["waves"])

    pos = _box_by_time(pb, ps, tf_big, tf_small) if box_mode=="time" else _box_by_pattern(pb, rb, ps, tf_big, tf_small)
    fig = _build_chart(symbol, tf_big, tf_small, pb, rb, ps, rs, frac, pos, box_mode)

    # Historical similarity analysis
    hist_matches, hist_current = _find_similar_history(pb, rb)
    hist_text = _history_report(hist_matches, hist_current, lang=page_lang) if hist_current else ""

    if page_lang == "ar":
        report = (_translate_report_ar(rb["report"]) + "\n\n" +
                  _translate_report_ar(rs["report"]) +
                  ("\n\n" + hist_text if hist_text else ""))
    else:
        report = (rb["report"] + "\n\n" + rs["report"] +
                  ("\n\n" + hist_text if hist_text else ""))

    phase_big = rb["current_phase"]
    phase_sml = rs["current_phase"]
    clr = {1:"info",2:"warning",3:"success",0:"secondary"}
    if page_lang == "ar":
        lbl = {1:"المرحلة 1 — اندفاع",2:"المرحلة 2 — تصحيح",
               3:"المرحلة 3 — استمرار",0:"جارٍ البحث..."}
    else:
        lbl = {1:"Phase 1 — Impulse",2:"Phase 2 — Correction",
               3:"Phase 3 — Continuation",0:"Searching..."}
    badge = html.Div([
        html.Span(f"{tf_big}: ", className="text-secondary small"),
        dbc.Badge(lbl.get(phase_big,"?"), color=clr.get(phase_big,"secondary"), className="me-2"),
        html.Br(),
        html.Span(f"{tf_small}: ", className="text-secondary small"),
        dbc.Badge(lbl.get(phase_sml,"?"), color=clr.get(phase_sml,"secondary")),
    ])
    at, ac = _make_alert(pos, tf_big, tf_small, symbol, box_mode)
    return fig, report, badge, at, ac, True


def _translate_report_ar(report):
    """Translate the ASCII pattern report to Arabic"""
    replacements = [
        ("PATTERN RECOGNITION REPORT",  "تقرير التعرف على الأنماط"),
        ("Total bars analysed",          "إجمالي الأشرطة المحللة"),
        ("Waves detected",               "الأمواج المكتشفة"),
        ("Impulses",                     "الاندفاعات"),
        ("Corrections",                  "التصحيحات"),
        ("Valid triples (W1W2W3)",        "الثلاثيات الصالحة (W1W2W3)"),
        ("VALID STRUCTURES:",            "الهياكل الصالحة:"),
        ("CURRENT PHASE",                "المرحلة الحالية"),
        ("Phase 1 — Impulse",            "المرحلة 1 — اندفاع"),
        ("Phase 2 — Correction",         "المرحلة 2 — تصحيح"),
        ("Phase 3 — Continuation",       "المرحلة 3 — استمرار"),
        ("Phase 3+ — Post-structure zone","المرحلة 3+ — منطقة ما بعد الهيكل"),
        ("No valid structure found",     "لم يُعثر على هيكل صالح"),
        ("Quality",                      "الجودة"),
        ("Phase=",                       "المرحلة="),
    ]
    for en, ar in replacements:
        report = report.replace(en, ar)
    return report


def _box_by_time(pb, ps, tf_big="1h", tf_small="15m"):
    n_big   = len(pb)
    n_small = len(ps)
    # Convert small TF bar count to equivalent big TF bars (time-correct)
    min_big   = TF_MINUTES.get(tf_big,   60)
    min_small = TF_MINUTES.get(tf_small, 15)
    # How many big bars equal the same time span as n_small small bars?
    n_small_in_big = int(n_small * min_small / min_big)
    # Cap: box never > 40% of big TF, never < 10 bars
    ratio = min(n_small_in_big / max(n_big, 1), 0.40)
    ratio = max(ratio, 10 / max(n_big, 1))
    b0 = max(0, int(n_big * (1 - ratio))); b1 = n_big - 1
    bp = pb[b0:]
    return {"mode":"time","box_start":b0,"box_end":b1,
            "box_min":float(np.min(bp)),"box_max":float(np.max(bp)),
            "box_color":ZOOM_COLOR,"wave_name":"time zone","wave_type":"time",
            "cur_price":float(pb[-1]),"cur_bar":n_big-1,
            "entry_signal":False,"label":"⏱ By Time"}


def _box_by_pattern(pb, rb, ps, tf_big="1h", tf_small="15m"):
    n_big=len(pb); n_small=len(ps)
    cur_bar=n_big-1; cur_price=float(pb[-1])
    # Convert small TF bar count to equivalent big TF bars (time-correct)
    min_big   = TF_MINUTES.get(tf_big,   60)
    min_small = TF_MINUTES.get(tf_small, 15)
    n_small_in_big = int(n_small * min_small / min_big)
    ratio = min(n_small_in_big / max(n_big, 1), 0.40)
    ratio = max(ratio, 10 / max(n_big, 1))
    ts=max(0,int(n_big*(1-ratio)))
    valid_big=  [t for t in rb["triples"] if t.is_valid]
    all_waves=  rb["waves"]
    wname="time zone"; wcolor=ZOOM_COLOR; wtype="unknown"
    method="by time"; entry=False
    if valid_big:
        last=max(valid_big, key=lambda t: t.w3.end.index)
        for wn,wv,wc in [("W1",last.w1,IMP_COLOR),("W2",last.w2,CORR_COLOR),("W3",last.w3,IMP_COLOR)]:
            ws=int(wv.start.index); we=int(wv.end.index)
            ext=max(int((we-ws)*0.35),3)
            if ws-ext<=cur_bar<=we+ext:
                wname=wn; wcolor=wc; wtype=wv.wave_type
                method=f"in {wn} of last structure"; entry=(wn=="W2"); break
    if wname=="time zone" and all_waves:
        for w in reversed(all_waves):
            if int(w.start.index)<=cur_bar<=int(w.end.index):
                wt=w.wave_type; wtype=wt
                wcolor=IMP_COLOR if wt=="impulse" else CORR_COLOR
                wname="Impulse" if wt=="impulse" else "Correction"
                method="current wave"; entry=(wt=="correction"); break
    if wname=="time zone" and all_waves:
        w=all_waves[-1]; wt=w.wave_type; wtype=wt
        wcolor=IMP_COLOR if wt=="impulse" else CORR_COLOR
        wname="post-structure"; method="after structure"
    b0=ts; b1=n_big-1; bp=pb[b0:]
    return {"mode":"pattern","box_start":b0,"box_end":b1,
            "box_min":float(np.min(bp)),"box_max":float(np.max(bp)),
            "box_color":wcolor,"wave_name":wname,"wave_type":wtype,
            "method":method,"cur_price":cur_price,"cur_bar":cur_bar,
            "entry_signal":entry,"label":f"📐 Pattern: [{wname}]"}


def _make_alert(pos, tf_big, tf_small, symbol, box_mode):
    price=pos.get("cur_price",0); wname=pos.get("wave_name","?")
    if pos.get("entry_signal") and box_mode=="pattern":
        return (f"🎯  ENTRY SIGNAL!  {symbol}  |  Small TF [{tf_small}] in [W2 — CORRECTION] "
                f"of [{tf_big}]  →  Wait W2 end, enter W3  |  Price: {price:,.2f}"), "danger"
    if box_mode=="time":
        return (f"⏱  {symbol}  |  By time mode  |  "
                f"Small TF [{tf_small}] = last N bars of [{tf_big}]  |  Price: {price:,.2f}"), "info"
    emoji={"W1":"🚀","W2":"🔄","W3":"📈"}.get(wname,"📍")
    color={"W1":"info","W2":"warning","W3":"success"}.get(wname,"info")
    return (f"{emoji}  {symbol}  |  Small TF [{tf_small}] in [{wname}] of [{tf_big}]  |  "
            f"Price: {price:,.2f}  ({pos.get('method','')})"), color


HIST_COLOR = "#A855F7"   # purple — similar past structures

def _find_similar_history(pb, rb, min_bars_after=10):
    """
    Find past W1W2W3 structures similar to the LAST valid structure.
    Similarity: |R_past - R_cur| < 0.15  AND  |S_past - S_cur| < 0.20
    Returns list of dicts with past structure info + what happened after.
    Minimum min_bars_after bars must exist after the structure to measure outcome.
    """
    valid = [t for t in rb["triples"] if t.is_valid]
    if len(valid) < 2:
        return [], None   # need at least 1 past + 1 current

    current = valid[-1]
    R_cur = current.correction_ratio
    S_cur = current.quality_score
    n     = len(pb)

    matches = []
    for t in valid[:-1]:   # all except the last (current)
        R_diff = abs(t.correction_ratio - R_cur)
        S_diff = abs(t.quality_score    - S_cur)
        if R_diff > 0.15 or S_diff > 0.20:
            continue

        # End of past structure = end of W3
        end_idx = int(t.w3.end.index)
        if end_idx + min_bars_after >= n:
            continue   # not enough bars after to measure

        # Measure what happened after: look ahead min_bars_after bars
        p_end   = float(pb[end_idx])
        p_after = float(pb[min(end_idx + min_bars_after, n-1)])
        pct_chg = (p_after - p_end) / p_end * 100

        # Also find max/min in lookahead window
        window = pb[end_idx : min(end_idx + min_bars_after*2, n)]
        p_max  = float(np.max(window))
        p_min  = float(np.min(window))

        matches.append({
            "w1_start": int(t.w1.start.index),
            "w3_end":   end_idx,
            "w1_start_price": float(pb[int(t.w1.start.index)]),
            "w3_end_price":   p_end,
            "R":   t.correction_ratio,
            "S":   t.quality_score,
            "pct_chg":  pct_chg,
            "p_max":    p_max,
            "p_min":    p_min,
            "went_up":  pct_chg > 0,
        })

    return matches, current


def _history_report(matches, current, lang="en"):
    """Build text summary of historical similarity analysis."""
    if not matches:
        return ""
    n      = len(matches)
    up     = sum(1 for m in matches if m["went_up"])
    down   = n - up
    pct_up = up / n * 100
    avg_up   = np.mean([m["pct_chg"] for m in matches if     m["went_up"]] or [0])
    avg_down = np.mean([m["pct_chg"] for m in matches if not m["went_up"]] or [0])

    R_c = current.correction_ratio
    S_c = current.quality_score

    if lang == "ar":
        lines = [
            "=" * 50,
            "  تحليل التشابه التاريخي (التاريخ يكرر نفسه)",
            "=" * 50,
            f"  البنية الحالية:  R={R_c:.3f}  S={S_c:.3f}",
            f"  بنى مشابهة في الماضي: {n}",
            "-" * 50,
            f"  صعود بعدها : {up}/{n}  ({pct_up:.0f}%)   متوسط +{avg_up:.1f}%",
            f"  هبوط بعدها : {down}/{n}  ({100-pct_up:.0f}%)   متوسط {avg_down:.1f}%",
            "-" * 50,
            f"  {'الاحتمال الأعلى: صعود 📈' if pct_up>=50 else 'الاحتمال الأعلى: هبوط 📉'}  ({max(pct_up,100-pct_up):.0f}%)",
            "=" * 50,
        ]
    elif lang == "ru":
        lines = [
            "=" * 50,
            "  АНАЛИЗ ИСТОРИЧЕСКОГО СХОДСТВА",
            "=" * 50,
            f"  Текущая структура:  R={R_c:.3f}  S={S_c:.3f}",
            f"  Похожих структур в прошлом: {n}",
            "-" * 50,
            f"  Рост после  : {up}/{n}  ({pct_up:.0f}%)   avg +{avg_up:.1f}%",
            f"  Падение после: {down}/{n}  ({100-pct_up:.0f}%)   avg {avg_down:.1f}%",
            "-" * 50,
            f"  {'Вероятнее рост 📈' if pct_up>=50 else 'Вероятнее падение 📉'}  ({max(pct_up,100-pct_up):.0f}%)",
            "=" * 50,
        ]
    else:
        lines = [
            "=" * 50,
            "  HISTORICAL SIMILARITY ANALYSIS",
            "=" * 50,
            f"  Current structure:  R={R_c:.3f}  S={S_c:.3f}",
            f"  Similar past structures found: {n}",
            "-" * 50,
            f"  Went UP after   : {up}/{n}  ({pct_up:.0f}%)   avg +{avg_up:.1f}%",
            f"  Went DOWN after : {down}/{n}  ({100-pct_up:.0f}%)   avg {avg_down:.1f}%",
            "-" * 50,
            f"  {'Most likely UP 📈' if pct_up>=50 else 'Most likely DOWN 📉'}  ({max(pct_up,100-pct_up):.0f}%)",
            "=" * 50,
        ]
    return "\n".join(lines)


def _build_chart(sym, tf_big, tf_small, pb, rb, ps, rs, frac, pos, box_mode):
    mode_label = "⏱ by time" if box_mode=="time" else "📐 by pattern"
    fig = make_subplots(rows=3, cols=1,
        row_heights=[0.45,0.35,0.20],
        subplot_titles=(
            f"🌍  {sym} {tf_big}  — Big TF (Mother pattern)  [{mode_label}]",
            f"🔬  {sym} {tf_small} — Small TF (inside big)",
            "📊  Wave Amplitudes — Fractality (Module 9)"),
        vertical_spacing=0.07)

    _add_layer(fig, 1, pb, rb, tf_big)
    _add_layer(fig, 2, ps, rs, tf_small)

    # ── Historical similarity markers (purple) ─────────────
    hist_matches, hist_current = _find_similar_history(pb, rb)
    for i, m in enumerate(hist_matches):
        x0 = m["w1_start"]; x1 = m["w3_end"]
        p0 = m["w1_start_price"]; p1 = m["w3_end_price"]
        pmin = float(np.min(pb[x0:x1+1])); pmax = float(np.max(pb[x0:x1+1]))
        pad  = (pmax - pmin) * 0.04
        arrow = "📈" if m["went_up"] else "📉"
        pct   = m["pct_chg"]
        # Shaded zone for past structure
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[pmin-pad, pmin-pad, pmax+pad, pmax+pad, pmin-pad],
            mode="lines", fill="toself",
            fillcolor="rgba(168,85,247,0.08)",
            line=dict(color=HIST_COLOR, width=1, dash="dot"),
            name=f"Similar #{i+1}" if i==0 else None,
            showlegend=(i==0),
            hoverinfo="skip",
        ), row=1, col=1)
        # Label: what happened after
        fig.add_annotation(
            x=x1, y=pmax+pad,
            text=f"<b>{arrow}{pct:+.1f}%</b>",
            showarrow=False,
            font=dict(size=9, color=HIST_COLOR),
            bgcolor="#1A1A2E", bordercolor=HIST_COLOR, borderwidth=1,
            row=1, col=1,
        )

    z0=pos["box_start"]; z1=pos["box_end"]
    bmin=pos["box_min"]; bmax=pos["box_max"]
    wcolor=pos["box_color"]; wname=pos["wave_name"]
    cur_bar=pos["cur_bar"]; cur_price=pos["cur_price"]
    pad=(bmax-bmin)*0.04; y0=bmin-pad; y1=bmax+pad

    # Box via closed Scatter
    fig.add_trace(go.Scatter(
        x=[z0,z1,z1,z0,z0], y=[y0,y0,y1,y1,y0],
        mode="lines", line=dict(color=wcolor,width=3),
        fill="toself", fillcolor=f"rgba({_hex_to_rgb(wcolor)},0.10)",
        name="YOU ARE HERE", showlegend=True, hoverinfo="skip",
    ), row=1, col=1)
    for xv in [z0,z1]:
        fig.add_trace(go.Scatter(x=[xv,xv],y=[y0,y1],mode="lines",showlegend=False,
            line=dict(color=wcolor,width=1.5,dash="dash"),hoverinfo="skip"), row=1, col=1)

    # Current price dot
    fig.add_trace(go.Scatter(x=[cur_bar],y=[cur_price],
        mode="markers+text",
        marker=dict(size=14,color=wcolor,line=dict(color="#FFF",width=2)),
        text=[f"  ◄ {cur_price:,.0f}"],textposition="middle right",
        textfont=dict(color="#FFF",size=11),showlegend=False), row=1, col=1)

    # Label above box
    mid_x=(z0+z1)/2
    fig.add_annotation(x=mid_x,y=y1,
        text=f"<b>🔍 YOU ARE HERE<br>{pos.get('label',wname)}</b>",
        showarrow=True,arrowhead=2,arrowsize=1.5,arrowwidth=2,arrowcolor=wcolor,
        ax=0,ay=-50,font=dict(size=11,color=wcolor),
        bgcolor="#1A1A2E",bordercolor=wcolor,borderwidth=2,row=1,col=1)

    # Entry signal
    if pos.get("entry_signal") and box_mode=="pattern":
        fig.add_trace(go.Scatter(x=[cur_bar],y=[cur_price],mode="markers",
            marker=dict(size=22,color="rgba(255,68,68,0.25)",symbol="circle",
                        line=dict(color="#FF4444",width=3)),
            name="🎯 Entry Signal",showlegend=True), row=1, col=1)
        fig.add_annotation(x=cur_bar,y=cur_price,
            text="<b>🎯 ENTRY POINT<br>Wait W2 end → enter W3</b>",
            showarrow=True,arrowhead=3,arrowsize=2,arrowwidth=2,arrowcolor="#FF4444",
            ax=70,ay=-50,font=dict(size=11,color="#FF4444"),
            bgcolor="#1A1A2E",bordercolor="#FF4444",borderwidth=2,row=1,col=1)

    # Current bar line
    fig.add_trace(go.Scatter(x=[cur_bar,cur_bar],
        y=[float(np.min(pb))*0.995,float(np.max(pb))*1.005],
        mode="lines",showlegend=False,hoverinfo="skip",
        line=dict(color="rgba(255,255,255,0.4)",width=1.5,dash="dot")), row=1, col=1)

    # Label on small TF
    fig.add_annotation(x=0.99,y=0.97,xref="x2 domain",yref="y2 domain",
        xanchor="right",yanchor="top",
        text=f"<b>⬆️ Zoom of [{wname}] ({mode_label})</b>",
        showarrow=False,font=dict(size=11,color=wcolor),
        bgcolor="#1A1A2E",bordercolor=wcolor,borderwidth=2)

    # ── 4 PHASE POINTS — always last 4 waves (right edge) ──
    all_waves_small = rs["waves"]
    n_ps = len(ps)
    if len(all_waves_small) >= 4:
        last4 = all_waves_small[-4:]
        points_4 = [
            (int(last4[0].start.index), "📍 Start"),
            (int(last4[1].end.index),   "📈 Rise"),
            (int(last4[2].end.index),   "➡️ Stable"),
            (int(last4[3].end.index),   "🏔 Peak"),
        ]
        ay_vals = [-45, -60, -45, -60]
        for (px, plabel), ay in zip(points_4, ay_vals):
            px = max(0, min(px, n_ps-1))
            fig.add_annotation(x=px, y=float(ps[px]),
                text=f"<b>{plabel}</b>",
                showarrow=True, arrowhead=2,
                arrowcolor="#FFD700", arrowwidth=1.5,
                ax=0, ay=ay,
                font=dict(size=10, color="#FFD700"),
                bgcolor="#1A1A2E", bordercolor="#FFD700", borderwidth=1,
                row=2, col=1)

    # Auto-zoom small TF — show exactly the time window of the big TF box
    n_big   = len(pb)
    n_small = len(ps)
    min_big   = TF_MINUTES.get(tf_big,   60)
    min_small = TF_MINUTES.get(tf_small, 15)
    # Box covers (z1-z0) big bars → convert to small bars
    box_big_bars   = max(z1 - z0, 1)
    box_minutes    = box_big_bars * min_big
    box_small_bars = int(box_minutes / min_small)
    # Add 10% padding on each side, minimum 40 bars
    pad_s  = max(int(box_small_bars * 0.10), 10)
    s_bars = box_small_bars + 2 * pad_s
    s_bars = max(s_bars, 40)
    s_end   = n_small - 1
    s_start = max(0, n_small - s_bars)
    if s_end > s_start + 5:
        fig.update_xaxes(range=[s_start, s_end], row=2, col=1)
        vp = ps[s_start:s_end + 1]
        if len(vp) > 0:
            vmin = float(np.min(vp)); vmax = float(np.max(vp))
            vpct = max((vmax - vmin) * 0.08, vmin * 0.0005)
            fig.update_yaxes(range=[vmin - vpct, vmax + vpct], row=2, col=1)

    # Amplitudes
    wb,ws2=rb["waves"],rs["waves"]
    if wb:
        fig.add_trace(go.Bar(x=[f"B{w.idx+1}" for w in wb],y=[w.amplitude for w in wb],
            name=tf_big,marker_color="#4361EE",opacity=0.8,
            text=["I" if w.wave_type=="impulse" else "C" for w in wb],
            textposition="outside"), row=3, col=1)
    if ws2:
        fig.add_trace(go.Bar(x=[f"S{w.idx+1}" for w in ws2],y=[w.amplitude for w in ws2],
            name=tf_small,marker_color="#F72585",opacity=0.8), row=3, col=1)

    coeff=frac["coefficient"]
    stab="Fractal stable ✅" if frac["stable"] else "Unstable ⚠️"
    lbl_f=f"coeff = {coeff:.4f}" if coeff else "Not enough waves"
    fig.add_annotation(xref="x3 domain",yref="y3 domain",x=0.99,y=0.95,
        xanchor="right",yanchor="top",text=f"<b>{lbl_f} | {stab}</b>",
        showarrow=False,bgcolor="#1A1A2E",bordercolor="#4361EE",borderwidth=1,
        font=dict(size=10,color="#E0E0FF"))

    fig.update_layout(template="plotly_dark",height=960,barmode="group",
        paper_bgcolor="#1A1A2E",plot_bgcolor="#1A1A2E",
        title=dict(text=(f"<b>{sym}</b>  —  Pattern Recognition  "
                         f"<span style='font-size:13px;color:#888'>"
                         f"[{tf_big}] + [{tf_small}]  |  Part resembles the whole</span>"),
                   font=dict(size=17,color="#E0E0FF")),
        legend=dict(orientation="h",y=-0.04,font=dict(color="#CCC")),
        margin=dict(l=50,r=50,t=65,b=40))
    return fig


def _hex_to_rgb(h):
    h=h.lstrip("#")
    return ",".join(str(int(h[i:i+2],16)) for i in (0,2,4))


def _add_layer(fig, row, prices, res, tf):
    df=res["dataframe"]; waves=res["waves"]; triples=res["triples"]
    extrema=res["extrema"]; phase=res["current_phase"]
    valid=[t for t in triples if t.is_valid]; x=list(range(len(prices)))

    fig.add_trace(go.Scatter(x=x,y=prices,mode="lines",name=f"Price {tf}",
        line=dict(color="#7B8CDE",width=1.3),opacity=0.5,
        showlegend=(row==1)),row=row,col=1)
    if "smooth" in df.columns:
        fig.add_trace(go.Scatter(x=x,y=df["smooth"].values,mode="lines",
            name="P̃(t)",showlegend=(row==1),
            line=dict(color="#B0B8FF",width=1.8,dash="dot")),row=row,col=1)

    for kind,sym2,color in [("max","triangle-up","#FF6B6B"),("min","triangle-down","#06D6A0")]:
        pts=[e for e in extrema if e.kind==kind]
        if pts:
            fig.add_trace(go.Scatter(x=[e.index for e in pts],y=[e.price for e in pts],
                mode="markers",showlegend=(row==1),name=kind,
                marker=dict(symbol=sym2,size=9,color=color)),row=row,col=1)

    for w in waves:
        c=IMP_COLOR if w.wave_type=="impulse" else CORR_COLOR
        fig.add_trace(go.Scatter(x=[w.start.index,w.end.index],y=[w.start.price,w.end.price],
            mode="lines+markers",showlegend=False,line=dict(color=c,width=2.5),
            marker=dict(size=5)),row=row,col=1)
        mx=(w.start.index+w.end.index)/2; my=min(w.start.price,w.end.price)
        fig.add_annotation(x=mx,y=my,
            text=f"<b>{'I' if w.wave_type=='impulse' else 'C'}</b>",
            showarrow=False,font=dict(size=9,color=c),row=row,col=1)

    for t in valid:
        x0,x1=t.w1.start.index,t.w3.end.index
        pc=PHASE_COLORS.get(t.phase,"#888")
        fig.add_vrect(x0=x0,x1=x1,fillcolor=pc,opacity=0.12,line_width=1,line_color=pc,row=row,col=1)
        for ww,lbl in [(t.w1,"W1"),(t.w2,"W2"),(t.w3,"W3")]:
            mx=(ww.start.index+ww.end.index)/2; my=(ww.start.price+ww.end.price)/2
            fig.add_annotation(x=mx,y=my,text=f"<b>{lbl}</b>",
                showarrow=True,arrowhead=0,arrowcolor=pc,
                font=dict(size=12,color=IMP_COLOR if lbl!="W2" else CORR_COLOR),row=row,col=1)
        fig.add_annotation(x=t.w2.end.index,y=t.w2.end.price,
            text=f"S={t.quality_score:.2f} R={t.correction_ratio:.2f}",
            showarrow=False,font=dict(size=9,color="#CCC"),
            bgcolor="#333",opacity=0.8,row=row,col=1)

    phase_info={1:("🚀 Phase 1 — Impulse","#00B4D8"),2:("🔄 Phase 2 — Correction","#FFB703"),
                3:("📈 Phase 3 — Continuation","#06D6A0"),0:("🔍 Searching...","#888")}
    txt,bg=phase_info.get(phase,("?","#888"))
    xref="x domain" if row==1 else f"x{row} domain"
    yref="y domain" if row==1 else f"y{row} domain"
    fig.add_annotation(x=0.01,y=0.97,xref=xref,yref=yref,xanchor="left",yanchor="top",
        text=f"<b>{txt}</b>  |  {len(valid)} structures",
        showarrow=False,bgcolor=bg,bordercolor="#FFF",borderwidth=1,
        font=dict(size=11,color="#111"))


def _load(symbol, period, interval):
    # Retry up to 3 times — Yahoo Finance sometimes blocks first request from server
    import time
    last_err = None
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(2 * attempt)  # wait 2s, 4s before retries
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                last_err = f"No data: {symbol} {interval} (attempt {attempt+1})"
                continue
            if hasattr(df.columns, "levels"):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            prices = df["Close"].dropna().values.flatten().astype(float)
            if len(prices) < 20:
                last_err = f"Too few bars: {len(prices)}"
                continue
            return prices
        except Exception as e:
            last_err = str(e)
            continue
    raise ValueError(last_err or f"No data: {symbol} {interval}")


def _error_fig(msg):
    fig=go.Figure()
    fig.add_annotation(text=f"❌ {msg}",xref="paper",yref="paper",
        x=0.5,y=0.5,showarrow=False,font=dict(size=16,color="#FF6B6B"))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#1A1A2E",plot_bgcolor="#1A1A2E")
    return fig


@app.callback(
    Output("report-collapse","is_open",allow_duplicate=True),
    Input("toggle-report","n_clicks"),
    State("report-collapse","is_open"),
    prevent_initial_call=True
)
def toggle_report(n, is_open):
    return not is_open

# ── Show/hide custom question field ───────────────────────
@app.callback(
    Output("custom-question-collapse", "is_open"),
    Input("ai-preset-mode", "value"),
    prevent_initial_call=False
)
def toggle_custom_question(mode):
    return mode == "custom"


# ══════════════════════════════════════════════════════════
# LOAD FREE MODELS FROM OPENROUTER
# ══════════════════════════════════════════════════════════
@app.callback(
    Output("ai-model",       "options"),
    Output("ai-model",       "value"),
    Output("models-status",  "children"),
    Input("btn-load-models", "n_clicks"),
    State("ai-api-key",      "value"),
    prevent_initial_call=True
)
def load_models(n_clicks, api_key):
    if not api_key or len(api_key) < 10:
        return AI_MODELS_DEFAULT, "none", "Enter API key first"
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15
        )
        if resp.status_code != 200:
            return AI_MODELS_DEFAULT, "none", f"Error {resp.status_code}: check your key"

        all_models = resp.json().get("data", [])

        vision_models = []
        text_models   = []
        for m in all_models:
            mid = m.get("id","")
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt","1") or "1")
            if prompt_price > 0:
                continue  # skip paid
            # check vision from OpenRouter data (reliable)
            arch = m.get("architecture", {})
            inp  = arch.get("input_modalities", []) or arch.get("modalities",{}).get("input",[])
            has_vision = "image" in str(inp).lower()
            name = m.get("name", mid)
            label = f"👁 {name}" if has_vision else f"📝 {name}"
            # Encode vision flag into value: "modelid|vision" or "modelid|text"
            val = f"{mid}|vision" if has_vision else f"{mid}|text"
            entry = {"label": label, "value": val}
            if has_vision:
                vision_models.append(entry)
            else:
                text_models.append(entry)

        options = (
            [{"label": "── Vision models (see image) ──", "value": "sep1", "disabled": True}]
            + vision_models
            + [{"label": "── Text only models ──", "value": "sep2", "disabled": True}]
            + text_models
        )

        if not vision_models and not text_models:
            return AI_MODELS_DEFAULT, "none", "No free models found"

        best = vision_models[0]["value"] if vision_models else (text_models[0]["value"] if text_models else "none")
        total = len(vision_models) + len(text_models)
        status = f"Loaded {total} free models ({len(vision_models)} with vision)"
        return options, best, status

    except Exception as e:
        return AI_MODELS_DEFAULT, "none", f"Error: {str(e)[:60]}"


# ══════════════════════════════════════════════════════════
# AI ANALYSIS CALLBACK
# ══════════════════════════════════════════════════════════
@app.callback(
    Output("ai-response", "children"),
    Output("ai-response", "color"),
    Input("btn-ai-analyse",   "n_clicks"),
    State("main-chart",        "figure"),
    State("ai-api-key",        "value"),
    State("ai-model",          "value"),
    State("ai-lang",           "value"),
    State("selected-symbol",   "data"),
    State("tf-selector",       "value"),
    State("tf-small-selector", "value"),
    State("position-alert",    "children"),
    State("ai-preset-mode",    "value"),
    State("ai-system-preset",  "value"),
    State("ai-custom-question","value"),
    prevent_initial_call=True
)
def ai_analyse(n_clicks, figure, api_key, model, lang,
               symbol, tf_name, tf_small_name, position_text,
               preset_mode, system_preset, custom_question):
    if not api_key or len(api_key) < 10:
        return "❌ No API key entered", "danger"
    if figure is None:
        return "❌ Run chart analysis first (press Analyse button)", "warning"
    try:
        return _ai_analyse_inner(
            figure, api_key, model, lang,
            symbol, tf_name, tf_small_name, position_text,
            preset_mode, system_preset, custom_question
        )
    except Exception as fatal_err:
        import traceback
        tb = traceback.format_exc()
        return f"❌ Fatal error:\n{str(fatal_err)}\n\n{tb[:800]}", "danger"


def _ai_analyse_inner(figure, api_key, model, lang,
                      symbol, tf_name, tf_small_name, position_text,
                      preset_mode, system_preset, custom_question):

    tf_big,   period_big   = TF_BIG_OPTIONS.get(tf_name,   ("1d","2y"))
    tf_small, period_small = TF_SMALL_OPTIONS.get(tf_small_name, ("1h","14d"))
    # Decode model id and vision flag (format: "modelid|vision" or "modelid|text")
    if "|" in (model or ""):
        model_id, vision_flag = model.rsplit("|", 1)
        is_vision = (vision_flag == "vision")
    else:
        model_id  = model or ""
        is_vision = any(x in model_id for x in ["vision","gemini","vl","llava"])
    model = model_id  # use clean model id for API call

    # 1. Chart -> PNG base64
    # On cloud servers (Render etc.) kaleido needs chromium — skip if unavailable
    img_content = []
    vision_status = "no"
    if is_vision:
        try:
            import plotly.io as pio
            fig_obj = go.Figure(figure)
            img_bytes = pio.to_image(fig_obj, format="png", width=1400, height=960, scale=1)
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            img_content = [{"type":"image_url",
                            "image_url":{"url":f"data:image/png;base64,{img_b64}"}}]
            vision_status = "yes"
        except Exception as kaleido_err:
            img_content = []
            is_vision = False
            vision_status = f"no (kaleido: {str(kaleido_err)[:60]})"

    # 2. Extract annotations from chart
    annot_lines = []
    if figure and "layout" in figure:
        for a in figure["layout"].get("annotations", [])[:20]:
            t = a.get("text","")
            if any(k in t for k in ["W1","W2","W3","Phase","YOU","S=","R="]):
                clean = t.replace("<b>","").replace("</b>","").replace("<br>"," | ")
                annot_lines.append(f"  - {clean}")
    annot_text = "\n".join(annot_lines) if annot_lines else "  - see image"

    # 3. Build system prompt (trading style)
    preset_mode   = preset_mode   or "overview"
    system_preset = system_preset or "trend"
    custom_question = custom_question or ""

    if lang == "ru":
        sys_prompt = AI_SYSTEM_PROMPTS_RU.get(system_preset,
                     AI_SYSTEM_PROMPTS_RU["trend"])
    elif lang == "ar":
        sys_prompt = AI_SYSTEM_PROMPTS_AR.get(system_preset,
                     AI_SYSTEM_PROMPTS_AR["trend"])
    else:
        sys_prompt = AI_SYSTEM_PROMPTS.get(system_preset,
                     AI_SYSTEM_PROMPTS["trend"])

    # Build mode-specific question
    if lang == "ru":
        MODE_QUESTIONS = {
            "entry": (
                "ЗАДАЧА: Найди лучшую точку входа прямо сейчас.\n"
                "Укажи: направление (long/short), точную цену входа, "
                "стоп-лосс, тейк-профит 1 и тейк-профит 2.\n"
                "Объясни почему именно здесь."
            ),
            "overview": (
                "ЗАДАЧА: Дай полный обзор рыночной ситуации.\n"
                "Опиши: общий тренд, текущую фазу цикла, "
                "ключевые уровни, что ожидать дальше."
            ),
            "risk": (
                "ЗАДАЧА: Оцени риски текущей позиции.\n"
                "Укажи: уровни стоп-лосс, сценарии против тренда, "
                "какие признаки говорят об опасности, что наблюдать."
            ),
            "correction": (
                "ЗАДАЧА: Проанализируй текущую коррекцию.\n"
                "Укажи: это коррекция или разворот? Где она закончится? "
                "Уровни поддержки для входа после коррекции."
            ),
            "custom": custom_question if custom_question else "Проанализируй график.",
        }
    elif lang == "ar":
        MODE_QUESTIONS = {
            "entry": (
                "المهمة: ابحث عن أفضل نقطة دخول الآن.\n"
                "حدد: الاتجاه (شراء/بيع)، سعر الدخول، "
                "وقف الخسارة، الهدف الأول والهدف الثاني.\n"
                "اشرح لماذا هنا بالذات."
            ),
            "overview": (
                "المهمة: أعطِ نظرة عامة كاملة على الوضع السوقي.\n"
                "صف: الاتجاه العام، المرحلة الحالية، "
                "المستويات الرئيسية، ماذا تتوقع لاحقاً."
            ),
            "risk": (
                "المهمة: قيّم مخاطر الوضع الحالي.\n"
                "حدد: مستويات وقف الخسارة، سيناريوهات عكس الاتجاه، "
                "ما هي علامات الخطر، ماذا تراقب."
            ),
            "correction": (
                "المهمة: حلّل التصحيح الحالي.\n"
                "حدد: هل هو تصحيح أم انعكاس؟ أين سينتهي؟ "
                "مستويات الدعم للدخول بعد التصحيح."
            ),
            "custom": custom_question if custom_question else "حلل الرسم البياني.",
        }
    else:
        MODE_QUESTIONS = {
            "entry": (
                "TASK: Find the best entry point right now.\n"
                "Specify: direction (long/short), exact entry price, "
                "stop-loss, take-profit 1 and take-profit 2.\n"
                "Explain why exactly here."
            ),
            "overview": (
                "TASK: Give a full market overview.\n"
                "Describe: overall trend, current cycle phase, "
                "key levels, what to expect next."
            ),
            "risk": (
                "TASK: Assess the risks of the current situation.\n"
                "Specify: stop-loss levels, counter-trend scenarios, "
                "warning signs, what to watch."
            ),
            "correction": (
                "TASK: Analyse the current correction.\n"
                "Specify: is this a correction or reversal? Where will it end? "
                "Support levels for entry after correction."
            ),
            "custom": custom_question if custom_question else "Analyse the chart.",
        }

    mode_question = MODE_QUESTIONS.get(preset_mode, MODE_QUESTIONS["overview"])

    # 3. Build prompt
    NL = "\n"
    if lang == "ru":
        q1 = f"1. Какой тренд на {tf_big}?"
        q2 = f"2. Где малый {tf_small} внутри большого? Красный квадрат правильно стоит?"
        q3 = "3. Паттерны совпадают на обоих ТФ или противоречат?"
        q4 = "4. Что делать прямо сейчас (войти/ждать/выйти)?"
        q5 = "5. Ключевые уровни."
        prompt = NL.join([
            sys_prompt,
            "",
            f"Инструмент: {symbol}",
            f"Таймфреймы: {tf_big} (большой) + {tf_small} (малый)",
            f"Позиция: {position_text}",
            "",
            "Данные с графика:",
            annot_text,
            "",
            mode_question,
            "",
            "Дополнительно ответь:",
            q1, q2, q3, q4, q5,
            "",
            "Отвечай конкретно, используй эмодзи.",
        ])
    elif lang == "ar":
        q1 = f"1. ما هو الاتجاه على {tf_big}?"
        q2 = f"2. أين يقع {tf_small} الصغير داخل الكبير؟ هل الإطار الأحمر في المكان الصحيح؟"
        q3 = "3. هل الأنماط على كلا الإطارين الزمنيين متوافقة أم متعارضة؟"
        q4 = "4. ماذا تفعل الآن؟ (دخول/انتظار/خروج)"
        q5 = "5. المستويات الرئيسية."
        prompt = NL.join([
            sys_prompt,
            "",
            f"الأداة: {symbol}",
            f"الإطارات الزمنية: {tf_big} (كبير) + {tf_small} (صغير)",
            f"الموقع الحالي: {position_text}",
            "",
            "بيانات من الرسم البياني:",
            annot_text,
            "",
            mode_question,
            "",
            "أجب أيضاً على:",
            q1, q2, q3, q4, q5,
            "",
            "كن محدداً واستخدم الرموز التعبيرية.",
        ])
    else:
        q1 = f"1. What is the trend on {tf_big}?"
        q2 = f"2. Where is small TF {tf_small} inside big TF? Is the red box correct?"
        q3 = "3. Do patterns on both TFs confirm or contradict each other?"
        q4 = "4. What to do right now (enter/wait/exit)?"
        q5 = "5. Key price levels."
        prompt = NL.join([
            sys_prompt,
            "",
            f"Instrument: {symbol}",
            f"Timeframes: {tf_big} (big) + {tf_small} (small)",
            f"Position: {position_text}",
            "",
            "Chart data:",
            annot_text,
            "",
            mode_question,
            "",
            "Also answer:",
            q1, q2, q3, q4, q5,
            "",
            "Be specific, use emojis.",
        ])

    # 4. Call OpenRouter
    user_content = img_content + [{"type":"text","text":prompt}]
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://pattern-recognition.app",
                "X-Title":       "Pattern Recognition System",
            },
            json={
                "model":    model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_content},
                ],
                "max_tokens":  2000,
                "temperature": 0.3,
            },
            timeout=90
        )
        if resp.status_code != 200:
            err = resp.json().get("error",{}).get("message", resp.text[:300])
            return f"API Error {resp.status_code}: {err}", "danger"

        data     = resp.json()
        ai_text  = data["choices"][0]["message"]["content"]
        used_mdl = data.get("model", model)

        is_ar = (lang == "ar")
        mode_label = next((m["label"] for m in AI_PRESET_MODES if m["value"]==preset_mode), "")
        style_label = next((m["label"] for m in AI_SYSTEM_PRESETS if m["value"]==system_preset), "")
        result = [
            html.Div([
                html.Span("🤖 ", style={"fontSize":"18px"}),
                html.Span("AI Analysis  ", className="fw-bold text-primary"),
                html.Span(f"({used_mdl})", className="text-secondary",
                          style={"fontSize":"11px"}),
                html.Span(f"  |  {mode_label}  |  {style_label}",
                          className="text-success",
                          style={"fontSize":"11px"}),
                html.Span(f"  |  vision={vision_status}",
                          className="text-secondary", style={"fontSize":"11px"}),
            ], className="mb-2 pb-1 border-bottom border-secondary",
               style={"direction":"ltr"}),
            html.Div(ai_text,
                     style={
                         "whiteSpace": "pre-wrap",
                         "lineHeight": "1.8",
                         "fontSize":   "14px",
                         "direction":  "rtl" if is_ar else "ltr",
                         "textAlign":  "right" if is_ar else "left",
                         "fontFamily": "Segoe UI, Tahoma, Arial, sans-serif",
                     }),
        ]
        return result, "dark"

    except requests.exceptions.Timeout:
        return "Timeout (90s). Try a faster model or check connection.", "warning"
    except Exception as e:
        return f"Error: {str(e)}", "danger"



if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Pattern Recognition System")
    print("  Open browser: http://127.0.0.1:8050")
    print("="*50 + "\n")
    app.run(debug=False, port=8050)











