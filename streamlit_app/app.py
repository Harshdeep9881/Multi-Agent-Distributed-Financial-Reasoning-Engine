import streamlit as st
import requests
import io
import logging
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from stock_symbols import ALL_STOCKS, CATEGORIES, get_stock_display_name, search_stocks

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    yf = None
    YFINANCE_AVAILABLE = False

# Add parent directory to path for importing agents
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from agents.prediction_agent import PredictionAgent
    from agents.graphing_agent import GraphingAgent
    PREDICTION_FEATURES_AVAILABLE = True
except Exception:
    PredictionAgent = None
    GraphingAgent = None
    PREDICTION_FEATURES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def currency_for_symbol(symbol):
    symbol = symbol.upper()
    if symbol.endswith("-USD"):
        return "USD"
    if symbol.endswith(".KS"):
        return "KRW"
    if symbol.endswith(".T"):
        return "JPY"
    if symbol.endswith(".HK"):
        return "HKD"
    if symbol.endswith(".NS"):
        return "INR"
    if symbol.endswith(".L"):
        return "GBP"
    if symbol.endswith(".DE") or symbol.endswith(".PA") or symbol.endswith(".AS") or symbol.endswith(".BR"):
        return "EUR"
    if symbol.endswith(".SW"):
        return "CHF"
    if symbol.endswith(".TO"):
        return "CAD"
    if symbol.endswith(".AX"):
        return "AUD"
    return "USD"


ETF_SYMBOLS = {
    symbol
    for category, symbols in CATEGORIES.items()
    if "ETF" in category.upper()
    for symbol in symbols
}
CRYPTO_SYMBOLS = {
    symbol
    for category, symbols in CATEGORIES.items()
    if "CRYPTO" in category.upper() or "₿" in category
    for symbol in symbols
}


def asset_icon(symbol):
    if symbol in CRYPTO_SYMBOLS or symbol.endswith("-USD"):
        return "₿"
    if symbol in ETF_SYMBOLS:
        return "📊"
    return "🏢"


def asset_label(symbol):
    company_name = ALL_STOCKS.get(symbol, "Custom Symbol")
    if company_name == "Custom Symbol":
        return f"{asset_icon(symbol)} {symbol}"
    return f"{asset_icon(symbol)} {symbol} - {company_name}"


@st.cache_data(ttl=60, show_spinner=False)
def get_live_prices(symbols):
    prices = {}
    if not YFINANCE_AVAILABLE or not symbols:
        return prices

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            fast_info = getattr(ticker, "fast_info", None)
            price = None
            if fast_info:
                price = fast_info.get("last_price") or fast_info.get("regularMarketPrice")
            if price is None:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price is not None:
                prices[symbol] = float(price)
        except Exception:
            continue
    return prices

# Initialize optional UI-side agents
prediction_agent = PredictionAgent() if PREDICTION_FEATURES_AVAILABLE else None
graphing_agent = GraphingAgent() if PREDICTION_FEATURES_AVAILABLE else None

# Determine the API URL based on environment
# On Streamlit Cloud: Set API_URL in Secrets
# Locally: Use localhost:8001
try:
    # Try to get from Streamlit secrets first (Streamlit Cloud)
    API_URL = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8001"))
except Exception:
    # Fallback to environment variable if secrets not available
    API_URL = os.environ.get("API_URL", "http://localhost:8001")

# Debug: Log the API URL being used
st.sidebar.text(f"API: {API_URL}")

st.title("🧠 Morning Market Brief Assistant")
st.markdown("""
**Professional market insights powered by AI - covering 460+ global stocks!**

Get comprehensive financial analysis on any stock from major exchanges worldwide:
🇺🇸 US Markets | 🌏 Asian Markets | 🌍 European Markets | 📊 ETFs | ₿ Crypto

Select stocks from the sidebar and ask your question below!
""")

# Data source info
st.info("""
📊 **Data Sources:** Real-time market data via Yahoo Finance API  
🤖 **AI Analysis:** Multi-agent system for portfolio risk assessment  
⚡ **Note:** On cloud deployments, news scraping may use enhanced fallback data enriched with market statistics
""")

# Display the current API endpoint (useful for debugging)
# st.sidebar.markdown(f"**API Endpoint:** `{API_URL}`")

# Stock selection feature
st.sidebar.header("📈 Stock Selection")
st.sidebar.markdown("**Select from 400+ global stocks, ETFs, and crypto!**")

# Selection method
selection_method = st.sidebar.radio(
    "Choose selection method:",
    ["Browse by Category", "Search by Name/Symbol", "Enter Custom Symbol"]
)

selected_stocks = []
selected_symbols = []

if selection_method == "Browse by Category":
    # Category-based selection
    selected_category = st.sidebar.selectbox("Select Category", list(CATEGORIES.keys()))
    
    # Get stocks for this category
    category_stocks = CATEGORIES[selected_category]
    selected_symbols = st.sidebar.multiselect(
        f"Multi-select assets from {selected_category}",
        category_stocks,
        format_func=asset_label,
        help="Select one or more assets to analyze"
    )
    selected_stocks = [get_stock_display_name(symbol) for symbol in selected_symbols]

elif selection_method == "Search by Name/Symbol":
    # Search-based selection
    search_query = st.sidebar.text_input(
        "🔍 Search stocks",
        placeholder="Type company name or symbol (e.g., Apple, MSFT, Tesla...)",
        help="Search across 400+ stocks"
    )
    
    if search_query:
        search_results = search_stocks(search_query)
        
        if search_results:
            st.sidebar.success(f"Found {len(search_results)} matches!")
            
            search_symbols = [symbol for symbol, _ in search_results[:20]]
            selected_symbols = st.sidebar.multiselect(
                "Multi-select assets from search results",
                search_symbols,
                format_func=asset_label,
                help="Select one or more assets"
            )
            selected_stocks = [get_stock_display_name(symbol) for symbol in selected_symbols]
        else:
            st.sidebar.warning("No stocks found. Try a different search term.")
            
elif selection_method == "Enter Custom Symbol":
    # Manual entry for any stock
    st.sidebar.info("💡 Enter any Yahoo Finance ticker symbol")
    custom_symbols_input = st.sidebar.text_input(
        "Enter symbol(s)",
        placeholder="e.g., AAPL, MSFT, TSM",
        help="Enter one or multiple symbols separated by commas"
    )
    
    if custom_symbols_input:
        # Parse comma-separated symbols
        selected_symbols = [s.strip().upper() for s in custom_symbols_input.split(",") if s.strip()]
        selected_stocks = [get_stock_display_name(symbol) for symbol in selected_symbols]
        
        st.sidebar.success(f"Selected: {', '.join(selected_symbols)}")

# Display selected stocks with details
if selected_symbols:
    st.sidebar.markdown("### ✅ Selected Assets:")
    sidebar_prices = get_live_prices(tuple(sorted(selected_symbols)))
    for symbol in selected_symbols:
        company_name = ALL_STOCKS.get(symbol, "Custom Symbol")
        price = sidebar_prices.get(symbol)
        price_text = f" ${price:,.2f}" if price is not None else ""
        if company_name != "Custom Symbol":
            st.sidebar.text(f"{asset_icon(symbol)} {symbol}{price_text} - {company_name}")
        else:
            st.sidebar.text(f"{asset_icon(symbol)} {symbol}{price_text}")
    
    st.sidebar.markdown(f"**Total: {len(selected_symbols)} stock(s)**")
    
    # Prepare query with selected stocks
    stock_query = f"Analyze {', '.join(selected_symbols)}. What's the current market situation and risk exposure?"
else:
    stock_query = ""
    st.sidebar.info("👆 Select stocks above to analyze")

# Show available stock count
st.sidebar.markdown("---")
st.sidebar.info(f"📊 Total available stocks: {len(ALL_STOCKS)}")
st.sidebar.caption("248 US | 49 Asian | 40 European | 98 ETFs | 20 Crypto")

# Quick reference guide
with st.sidebar.expander("📖 Symbol Format Guide"):
    st.markdown("""
    **Stock Symbol Formats:**
    - 🇺🇸 US: `AAPL`, `MSFT`, `GOOGL`
    - 🇰🇷 Korea: `005930.KS` (Samsung)
    - 🇯🇵 Japan: `7203.T` (Toyota)
    - 🇨🇳 Hong Kong: `0700.HK` (Tencent)
    - 🇮🇳 India: `RELIANCE.NS`
    - 🇬🇧 UK: `BP.L`
    - 🇩🇪 Germany: `SAP.DE`
    - 🇫🇷 France: `MC.PA`
    - 🇨🇭 Switzerland: `NESN.SW`
    - 🇨🇦 Canada: `SHOP.TO`
    - 🇦🇺 Australia: `CBA.AX`
    - ₿ Crypto: `BTC-USD`, `ETH-USD`
    
    **Tips:**
    - Most US stocks use simple tickers
    - International stocks have country suffixes
    - Search by company name if unsure
    """)

# Guided workflow
st.markdown("### 🔎 1. Select Assets")
if selected_symbols:
    live_prices = get_live_prices(tuple(sorted(selected_symbols)))
    asset_lines = []
    for symbol in selected_symbols:
        price = live_prices.get(symbol)
        if price is None:
            asset_lines.append(f"{asset_icon(symbol)} `{symbol}`")
        else:
            asset_lines.append(f"{asset_icon(symbol)} `{symbol}` ${price:,.2f}")
    st.success("Selected assets: " + " | ".join(asset_lines))
else:
    st.warning("No assets selected yet. Use the multi-select dropdown in the sidebar.")

st.markdown("### 🧠 2. Choose What You Want")
analysis_mode = st.radio(
    "Outcome",
    ["📊 Market Summary", "⚠️ Risk Breakdown", "📈 Compare Performance", "🎤 Ask by Voice"],
    horizontal=True,
)

query_templates = {
    "📊 Market Summary": "Give me a concise market brief for the selected symbols.",
    "⚠️ Risk Breakdown": "Analyze risk exposure, volatility, and concentration for my selected symbols.",
    "📈 Compare Performance": "Compare valuation, price trend, and volume for the selected symbols.",
    "🎤 Ask by Voice": "Prepare analysis for a voice query on the selected symbols.",
}

st.markdown("### 3. Build Your Question")
with st.expander("Quick starters"):
    example_keys = ["None"] + list(query_templates.keys())
    chosen = st.selectbox("Prefill query from mode", example_keys)
    if st.button("Apply Starter", use_container_width=True):
        if chosen != "None":
            st.session_state["query_input"] = query_templates[chosen]

default_query = stock_query if stock_query else "What's our risk exposure in tech stocks today?"
if "query_input" not in st.session_state:
    st.session_state["query_input"] = default_query

query = st.text_input(
    "Enter your market analysis question:",
    key="query_input",
    help="Ask about specific stocks, market trends, or portfolio risk",
)

with st.expander("What this run will execute"):
    st.markdown(
        """
        - Backend health check
        - Retrieval pipeline: market data + news context + retriever
        - Analysis pipeline: risk exposure + earnings + language summary
        - Capability checks: optional earnings endpoint and local prediction modules
        """
    )
    if PREDICTION_FEATURES_AVAILABLE:
        st.success("UI prediction/graph modules are available.")
    else:
        st.warning("Prediction/graph modules are unavailable in this build. Core analysis still works.")


PROGRESS_STEPS = [
    ("fetch", "🔄 Fetching Market Data (Yahoo Finance)"),
    ("scrape", "📰 Scraping News Articles"),
    ("context", "📚 Retrieving Relevant Context"),
    ("risk", "📊 Calculating Risk Exposure"),
    ("summary", "🤖 Generating AI Summary"),
]


def render_progress_tracker(states, placeholder):
    rows = []
    for step_id, label in PROGRESS_STEPS:
        state = states.get(step_id, "pending")
        if state == "done":
            rows.append(f"<div style='color:#16a34a;'>✅ {label}</div>")
        elif state == "running":
            rows.append(f"<div style='color:#1d4ed8;'>{label}</div>")
        elif state == "failed":
            rows.append(f"<div style='color:#dc2626;'>❌ {label}</div>")
        else:
            rows.append(f"<div style='color:#6b7280;'>⏳ {label}</div>")
    placeholder.markdown("<br>".join(rows), unsafe_allow_html=True)


if st.button("Run Analysis", type="primary"):
    with st.spinner("Processing query..."):
        try:
            progress_states = {step_id: "pending" for step_id, _ in PROGRESS_STEPS}
            progress_placeholder = st.empty()
            render_progress_tracker(progress_states, progress_placeholder)

            st.info(f"Connecting to backend: {API_URL}")
            health_check = requests.get(f"{API_URL}/", timeout=10)
            if health_check.status_code != 200:
                st.error(f"FastAPI server not reachable: {health_check.status_code} - {health_check.text}")
            else:
                st.success("Backend is healthy. Running pipeline...")
                params = {"query": query}
                if selected_symbols:
                    params["symbols"] = ",".join(selected_symbols)

                progress_states["fetch"] = "running"
                render_progress_tracker(progress_states, progress_placeholder)
                retrieve_response = requests.get(f"{API_URL}/retrieve/retrieve", params=params, timeout=30)
                if retrieve_response.status_code != 200:
                    progress_states["fetch"] = "failed"
                    render_progress_tracker(progress_states, progress_placeholder)
                    st.error(f"Retrieval failed with status {retrieve_response.status_code}")
                    st.code(retrieve_response.text[:500])
                else:
                    progress_states["fetch"] = "done"
                    progress_states["scrape"] = "done"
                    progress_states["context"] = "done"
                    progress_states["risk"] = "running"
                    render_progress_tracker(progress_states, progress_placeholder)
                    retrieve_data = retrieve_response.json()
                    if "error" in retrieve_data:
                        progress_states["risk"] = "failed"
                        render_progress_tracker(progress_states, progress_placeholder)
                        st.error(retrieve_data["error"])
                        if "suggestion" in retrieve_data:
                            st.info(retrieve_data["suggestion"])
                    else:
                        analyze_response = requests.post(
                            f"{API_URL}/analyze/analyze",
                            json={"data": retrieve_data},
                            timeout=60,
                        )
                        if analyze_response.status_code != 200:
                            progress_states["risk"] = "failed"
                            render_progress_tracker(progress_states, progress_placeholder)
                            st.error(f"Analysis failed with status {analyze_response.status_code}")
                            st.code(analyze_response.text[:500])
                        else:
                            progress_states["risk"] = "done"
                            progress_states["summary"] = "running"
                            render_progress_tracker(progress_states, progress_placeholder)
                            analyze_data = analyze_response.json()
                            if "error" in analyze_data:
                                progress_states["summary"] = "failed"
                                render_progress_tracker(progress_states, progress_placeholder)
                                st.error(f"Analysis error: {analyze_data['error']}")
                            else:
                                progress_states["summary"] = "done"
                                render_progress_tracker(progress_states, progress_placeholder)
                                summary = analyze_data.get("summary", "")
                                contexts = retrieve_data.get("context", [])
                                markets = retrieve_data.get("market_data", {})
                                symbols = retrieve_data.get("symbols", [])
                                agents_used = analyze_data.get("agents_used", retrieve_data.get("agents_used", []))
                                timing_breakdown_ms = analyze_data.get("timing_breakdown_ms", retrieve_data.get("timing_breakdown_ms", {}))

                                fallback_context = any(
                                    ("market data retrieved" in str(ctx).lower()) or ("analysis:" in str(ctx).lower())
                                    for ctx in contexts
                                )
                                dummy_summary = ("2023" in summary and "2024" in summary and "10.5" in summary)

                                context_text = " ".join([str(ctx).lower() for ctx in contexts])
                                cached_context = "cache" in context_text or "cached" in context_text
                                if cached_context:
                                    quality_state = "cached"
                                elif fallback_context or dummy_summary:
                                    quality_state = "fallback"
                                else:
                                    quality_state = "live"

                                quality_badges = {
                                    "live": "<span style='background:#dcfce7;color:#166534;padding:4px 10px;border-radius:999px;font-weight:600;'>🟢 Live Data</span>",
                                    "fallback": "<span style='background:#fef9c3;color:#854d0e;padding:4px 10px;border-radius:999px;font-weight:600;'>🟡 Fallback Data</span>",
                                    "cached": "<span style='background:#fee2e2;color:#991b1b;padding:4px 10px;border-radius:999px;font-weight:600;'>🔴 Cached Data</span>",
                                }

                                news_coverage = sum(
                                    1
                                    for ctx in contexts
                                    if any(token in str(ctx).lower() for token in ["news", "article", "source", "http", "www"])
                                )
                                if news_coverage == 0:
                                    news_coverage = len(contexts)

                                confidence_score = 70 + min(len(symbols) * 2, 10) + min(len(contexts) * 3, 15)
                                if quality_state == "live":
                                    confidence_score += 5
                                elif quality_state == "fallback":
                                    confidence_score -= 12
                                else:
                                    confidence_score -= 20
                                if dummy_summary:
                                    confidence_score -= 8
                                confidence_score = max(5, min(95, confidence_score))

                                freshness_label = {
                                    "live": "Real-time",
                                    "fallback": "Partial fallback",
                                    "cached": "Cached",
                                }[quality_state]

                                st.subheader("Results Overview")
                                c1, c2 = st.columns(2)
                                c1.metric("Symbols Analyzed", len(symbols))
                                c2.metric("Context Items", len(contexts))
                                st.markdown(quality_badges[quality_state], unsafe_allow_html=True)

                                m1, m2, m3 = st.columns(3)
                                m1.metric("AI Confidence", f"{confidence_score}%")
                                m2.metric("Data Freshness", freshness_label)
                                m3.metric("News Coverage", f"{news_coverage} articles analyzed")

                                if agents_used:
                                    st.markdown("#### 🧠 Agents Used")
                                    display_name_map = {
                                        "API Agent": "Market Data Agent",
                                        "API Agent (Earnings)": "Earnings Agent",
                                        "Scraping Agent": "News Agent",
                                        "Retriever Agent": "Context Retriever Agent",
                                        "Analysis Agent": "Risk Analysis Agent",
                                        "Language Agent": "Language Agent",
                                    }
                                    for agent in agents_used:
                                        raw_name = agent.get("name", "Unknown Agent")
                                        display_name = display_name_map.get(raw_name, raw_name)
                                        status = agent.get("status", "success")
                                        time_ms = agent.get("time_ms")
                                        if status in ("success", "cache_hit"):
                                            icon = "✓"
                                            suffix = " (Cached)" if status == "cache_hit" else ""
                                        elif status == "fallback_used":
                                            icon = "⚠"
                                            suffix = " (Fallback Mode)"
                                        else:
                                            icon = "✗"
                                            suffix = " (Failed)"
                                        if time_ms is not None:
                                            st.write(f"{icon} {display_name}{suffix} - {time_ms} ms")
                                        else:
                                            st.write(f"{icon} {display_name}{suffix}")

                                if timing_breakdown_ms:
                                    total_ms = timing_breakdown_ms.get(
                                        "total_response_time_end_to_end",
                                        timing_breakdown_ms.get("total_response_time"),
                                    )
                                    market_ms = timing_breakdown_ms.get("market_data", 0)
                                    news_ms = timing_breakdown_ms.get("news", 0)
                                    ai_analysis_ms = (
                                        timing_breakdown_ms.get("risk_analysis", 0)
                                        + timing_breakdown_ms.get("language", 0)
                                        + timing_breakdown_ms.get("earnings", 0)
                                    )
                                    st.markdown("#### ⏱️ Execution Timing")
                                    if total_ms is None:
                                        total_ms = market_ms + news_ms + ai_analysis_ms
                                    t1, t2, t3, t4 = st.columns(4)
                                    t1.metric("Total Response Time", f"{(total_ms / 1000):.1f}s")
                                    t2.metric("Market Data", f"{(market_ms / 1000):.1f}s")
                                    t3.metric("News", f"{(news_ms / 1000):.1f}s")
                                    t4.metric("AI Analysis", f"{(ai_analysis_ms / 1000):.1f}s")

                                result_tabs = st.tabs(
                                    ["📊 Summary", "⚠️ Risk", "📰 News", "📈 Charts", "🔮 Forecast"]
                                )

                                with result_tabs[0]:
                                    st.markdown("#### Executive Snapshot")
                                    summary_lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
                                    top_lines = summary_lines[:5] if summary_lines else ["No summary text generated."]
                                    for line in top_lines:
                                        st.write(f"- {line}")
                                    st.markdown("#### Full Brief")
                                    st.markdown(summary)

                                with result_tabs[1]:
                                    st.markdown("#### Position Metrics from Retrieved Market Data")
                                    if not markets:
                                        st.info("No market data available.")
                                    else:
                                        risk_rows = []
                                        for sym, records in markets.items():
                                            if isinstance(records, list) and records:
                                                df = pd.DataFrame(records)
                                                latest_close = df["Close"].iloc[-1] if "Close" in df.columns else None
                                                avg_volume = df["Volume"].mean() if "Volume" in df.columns else None
                                                risk_rows.append(
                                                    {
                                                        "Symbol": sym,
                                                        "Currency": currency_for_symbol(sym),
                                                        "Latest Close": f"{latest_close:,.2f}" if pd.notna(latest_close) else "N/A",
                                                        "Avg Volume": f"{avg_volume:,.0f}" if pd.notna(avg_volume) else "N/A",
                                                    }
                                                )
                                        if risk_rows:
                                            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True)
                                        else:
                                            st.info("Could not parse market records for risk table.")
                                    if fallback_context:
                                        st.warning("Fallback context text was used for part of the analysis.")
                                    if dummy_summary:
                                        st.warning("Fallback dummy earnings defaults were detected in the brief.")

                                with result_tabs[2]:
                                    st.markdown("#### News and Retrieved Context")
                                    if not contexts:
                                        st.info("No context documents returned.")
                                    else:
                                        for i, ctx in enumerate(contexts, 1):
                                            st.markdown(f"**Item {i}:** {ctx}")

                                with result_tabs[3]:
                                    st.markdown("#### Price Charts")
                                    if not markets:
                                        st.info("No market data available for charting.")
                                    else:
                                        chart_symbols = [sym for sym, records in markets.items() if isinstance(records, list) and records]
                                        if not chart_symbols:
                                            st.info("Could not parse market records for charts.")
                                        else:
                                            chart_symbol = st.selectbox("Select symbol for chart", chart_symbols)
                                            chart_df = pd.DataFrame(markets[chart_symbol])
                                            if "Close" in chart_df.columns:
                                                st.line_chart(chart_df[["Close"]], use_container_width=True)
                                            else:
                                                st.info("Close price series unavailable for this symbol.")

                                with result_tabs[4]:
                                    st.markdown("#### Forecast Outlook")
                                    if not markets:
                                        st.info("Insufficient market data for a directional forecast.")
                                    else:
                                        forecast_rows = []
                                        for sym, records in markets.items():
                                            if isinstance(records, list) and records:
                                                df = pd.DataFrame(records)
                                                if "Close" in df.columns and len(df["Close"]) >= 2:
                                                    first_close = df["Close"].iloc[0]
                                                    last_close = df["Close"].iloc[-1]
                                                    if pd.notna(first_close) and first_close != 0 and pd.notna(last_close):
                                                        change_pct = ((last_close - first_close) / first_close) * 100
                                                        bias = "Bullish" if change_pct > 1 else "Bearish" if change_pct < -1 else "Neutral"
                                                        forecast_rows.append(
                                                            {
                                                                "Symbol": sym,
                                                                "Trend (period)": f"{change_pct:+.2f}%",
                                                                "Outlook Bias": bias,
                                                            }
                                                        )
                                        if forecast_rows:
                                            st.dataframe(pd.DataFrame(forecast_rows), use_container_width=True)
                                        else:
                                            st.info("Not enough close-price history to estimate a forecast trend.")
                                    if PREDICTION_FEATURES_AVAILABLE:
                                        st.caption("Advanced prediction modules are available in this environment.")
                                    else:
                                        st.caption("Advanced prediction modules are disabled in this build.")

                                developer_mode = st.toggle("🔍 Developer Mode (Advanced Users)", value=False)
                                if developer_mode:
                                    with st.expander("Advanced Pipeline Data", expanded=True):
                                        st.json(
                                            {
                                                "query": retrieve_data.get("query"),
                                                "symbols": symbols,
                                                "market_keys": list(markets.keys()),
                                                "context_count": len(contexts),
                                                "data_quality": quality_state,
                                                "confidence_score": confidence_score,
                                                "data_freshness": freshness_label,
                                                "news_coverage": news_coverage,
                                                "agents_used": agents_used,
                                                "timing_breakdown_ms": timing_breakdown_ms,
                                            }
                                        )
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend API. Start the FastAPI orchestrator and retry.")
            logger.error(f"Connection error: Failed to connect to FastAPI server at {API_URL}")
        except Exception as e:
            st.error(f"Failed to process query: {str(e)}")
            logger.error(f"Exception in query processing: {str(e)}")

# Audio query input (optional for voice I/O)
st.subheader("Step 4 (Optional): Upload an Audio Query (WAV/MP3)")
audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    with st.spinner("Processing audio query..."):
        try:
            files = {"audio": (audio_file.name, audio_file, "audio/wav")}
            
            # If we have selected symbols, add them as form data
            data = {}
            if selected_symbols:
                data = {"symbols": ",".join(selected_symbols)}
                
            response = requests.post(f"{API_URL}/process_query", files=files, data=data, stream=True)
            logger.info(f"Audio process_query response status: {response.status_code}")
            
            if response.status_code == 200:
                audio_bytes = io.BytesIO(response.content)
                st.audio(audio_bytes, format="audio/mp3")
                st.success("Audio query processed successfully!")
            else:
                try:
                    error_msg = response.json().get("error", "Unknown error")
                    st.error(f"Audio processing error: {error_msg}")
                    logger.error(f"Audio processing error: {error_msg}")
                except:
                    st.error(f"Audio processing failed with status {response.status_code}")
                    logger.error(f"Audio processing failed with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error(f"FastAPI server is trying to connect to Render services. If it takes long, try running it locally.")
            logger.error(f"Connection error: Failed to connect to FastAPI server at {API_URL}")
        except Exception as e:
            st.error(f"Failed to process audio query: {str(e)}")
            logger.error(f"Exception in audio query processing: {str(e)}")

# Add a section to explain how the application works
with st.expander("ℹ️ How this application works"):
    st.write("""
    ### Architecture
    
    This application uses a **separated frontend-backend architecture**:
    
    1. **Streamlit Frontend** (This interface):
       - User-friendly interface for stock selection and queries
       - Communicates with the FastAPI backend via REST API
       - Displays analysis results and market briefs
    
    2. **FastAPI Backend** (Orchestrator):
       - Runs independently on port 8001
       - Orchestrates multiple AI agents for data processing
       - Handles API requests from the frontend
    
    3. **AI Agents**:
       - **API Agent**: Fetches market data via yfinance
       - **Scraping Agent**: Crawls news using newspaper3k  
       - **Retriever Agent**: Indexes and retrieves relevant data
       - **Analysis Agent**: Calculates portfolio risk exposure
       - **Language Agent**: Generates comprehensive narratives
       - **Voice Agent**: Handles speech-to-text and text-to-speech
    
    ### Data Flow:
    
    ```
    User Query → Streamlit → FastAPI Backend → AI Agents → Analysis → Response → Streamlit → User
    ```
    
    ### Features:
    
    - ✅ **460+ Global Stocks**: US, Asia, Europe, Canada, Australia
    - ✅ **ETFs & Indices**: SPY, QQQ, S&P 500, NASDAQ, and more
    - ✅ **Cryptocurrencies**: Bitcoin, Ethereum, and major altcoins
    - ✅ **Real-time Data**: Via Yahoo Finance API
    - ✅ **News Analysis**: Scrapes and analyzes market news
    - ✅ **Risk Assessment**: Portfolio exposure analysis
    - ✅ **AI-Powered Briefs**: Comprehensive market narratives
    
    When you ask a question, the system:
    1. Fetches real-time market data
    2. Retrieves relevant news articles
    3. Calculates portfolio risk and exposure
    4. Generates earnings analysis
    5. Creates a comprehensive market brief
    """)

# Add a section to explain how to debug the application
with st.expander("🔧 Troubleshooting"):
    st.write("""
    ### Common issues and solutions:
    
    1. **FastAPI server not reachable**: Make sure the FastAPI server is running on the right port with:
       ```
       uvicorn orchestrator.orchestrator:app --host 0.0.0.0 --port 8001
       ```
       
    2. **Running in Streamlit Cloud**: Both services must be running. Make sure the FastAPI orchestrator 
       is started in the Dockerfile or defined in your cloud deployment.
       
    3. **Internal Server Error (500)**: Check the logs of the FastAPI server for more details.
    
    4. **JSON serialization errors**: These often happen with pandas DataFrames. The updated code should handle this.
    
    5. **Empty or incorrect responses**: The system now has fallback data to ensure you always get a response.
    
    6. **Stock not recognized**: Try using the ticker symbol directly (e.g., AAPL instead of Apple) in your query.
    """)
