
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string
)

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

OUTPUTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

app = Flask(__name__)

# ============================================================
# 2. PIPELINE EXECUTION ORDER
# ============================================================


PIPELINE = [
    ("Data Generation", SRC / "data_generation.py"),
    ("Data Processing", SRC / "data_processing.py"),
    ("Feature Engineering", SRC / "feature_engineering.py"),
    ("Demand Forecasting", SRC / "demand_forecasting.py"),
    ("Model Evaluation", SRC / "evaluation.py"),
    ("Opportunity Detection", SRC / "opportunity_detection.py"),
    ("Inventory Optimization", SRC / "inventory_optimization.py"),
    ("Scenario Analysis", SRC / "scenario_generation.py"),
]

# Expected output files
OUTPUT_FILES = {
    "forecast": "forecast_results.csv",
    "opportunities": "opportunity_results.csv",
    "optimizer": "optimal_procurement_plan.csv",
    "scenarios": "scenario_results.csv",
}

# Application state
pipeline_logs = []
pipeline_status = "Not started"
last_run = None


# ============================================================
# 3. RUN ALL PYTHON SCRIPTS
# ============================================================

def run_pipeline():

    global pipeline_logs
    global pipeline_status
    global last_run

    pipeline_logs = []
    pipeline_status = "Running"

    print("\n" + "=" * 60)
    print("TRAVELPULSE EXECUTION ENGINE")
    print("=" * 60)

    for index, (name, script) in enumerate(PIPELINE, start=1):

        print(f"\n[{index}/{len(PIPELINE)}] Running {name}")

        if not script.exists():

            message = f"Script not found: {script}"

            pipeline_logs.append({
                "name": name,
                "status": "FAILED",
                "output": message
            })

            pipeline_status = "Failed"
            print(message)

            return False

        try:

            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=900,
                check=True,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1"
                }
            )

            pipeline_logs.append({
                "name": name,
                "status": "SUCCESS",
                "output": result.stdout[-5000:]
            })

            print(f"SUCCESS: {name}")

        except subprocess.CalledProcessError as error:

            output = (
                (error.stdout or "")
                + "\n"
                + (error.stderr or "")
            )

            pipeline_logs.append({
                "name": name,
                "status": "FAILED",
                "output": output[-7000:]
            })

            pipeline_status = "Failed"

            print(f"FAILED: {name}")
            print(output)

            return False

        except subprocess.TimeoutExpired:

            pipeline_logs.append({
                "name": name,
                "status": "FAILED",
                "output": "Script exceeded 15 minutes."
            })

            pipeline_status = "Failed"

            return False

    pipeline_status = "Completed"
    last_run = datetime.now().strftime("%d %b %Y, %I:%M:%S %p")

    print("\nALL PIPELINE SCRIPTS COMPLETED")

    return True


# ============================================================
# 4. LOAD GENERATED CSV FILES
# ============================================================

def load_data():

    data = {}

    for key, filename in OUTPUT_FILES.items():

        path = OUTPUTS / filename

        if path.exists():

            try:
                data[key] = pd.read_csv(path)

            except Exception as error:

                print(f"Error loading {filename}: {error}")
                data[key] = pd.DataFrame()

        else:

            data[key] = pd.DataFrame()

    return data


# ============================================================
# 5. DASHBOARD HELPERS
# ============================================================

def get_metric(df, column, operation="sum"):

    if df.empty or column not in df.columns:
        return 0

    if operation == "unique":
        return df[column].nunique()

    if operation == "mean":
        return df[column].mean()

    return df[column].sum()


def create_chart(fig):

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )


def create_table(df):

    if df.empty:
        return "<p class='empty'>No data available.</p>"

    return df.to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=True
    )


# ============================================================
# 6. DASHBOARD HTML
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>

<title>TravelPulse | Intelligence Platform</title>

<meta name="viewport" content="width=device-width, initial-scale=1">

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
    background: #0b1120;
    color: #e5e7eb;
}

.sidebar {
    position: fixed;
    width: 245px;
    height: 100vh;
    background: #111827;
    padding: 28px 18px;
    border-right: 1px solid #263247;
}

.logo {
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 5px;
}

.logo span {
    color: #38bdf8;
}

.subtitle {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 40px;
}

.nav-title {
    color: #64748b;
    font-size: 11px;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}

.sidebar a {
    display: block;
    text-decoration: none;
    color: #cbd5e1;
    padding: 13px;
    border-radius: 8px;
    margin-bottom: 7px;
    font-size: 14px;
}

.sidebar a:hover,
.sidebar a.active {
    background: #1e3a5f;
    color: #38bdf8;
}

.main {
    margin-left: 245px;
    padding: 35px;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 15px;
    margin-bottom: 35px;
}

h1 {
    font-size: 29px;
    margin: 0;
}

.description {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 9px;
}

.status {
    background: #172b3b;
    color: #38bdf8;
    padding: 10px 15px;
    border-radius: 8px;
    font-size: 12px;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    margin-bottom: 25px;
}

.card {
    background: #151f30;
    border: 1px solid #263247;
    border-radius: 12px;
    padding: 22px;
    min-width: 0;
}

.metric-label {
    font-size: 13px;
    color: #94a3b8;
}

.metric-value {
    font-size: 27px;
    font-weight: 750;
    margin-top: 13px;
    color: #f8fafc;
    overflow-wrap: anywhere;
}

.section {
    margin-top: 25px;
}

.section-title {
    font-size: 18px;
    margin-bottom: 17px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
}

.chart {
    background: white;
    border-radius: 12px;
    padding: 12px;
    min-width: 0;
    overflow: hidden;
}

.table-wrap {
    overflow-x: auto;
    background: #151f30;
    border: 1px solid #263247;
    border-radius: 10px;
    padding: 12px;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    white-space: nowrap;
}

.data-table th {
    color: #38bdf8;
    text-align: left;
    padding: 13px;
    border-bottom: 1px solid #334155;
}

.data-table td {
    padding: 12px;
    border-bottom: 1px solid #263247;
}

.empty {
    color: #94a3b8;
    padding: 20px;
}

.button {
    background: #0284c7;
    color: white;
    padding: 12px 18px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
}

.button:hover {
    background: #0369a1;
}

.log {
    background: #111827;
    padding: 17px;
    margin-bottom: 12px;
    border-radius: 9px;
    border: 1px solid #263247;
}

.success {
    color: #4ade80;
}

.failed {
    color: #f87171;
}

pre {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    color: #cbd5e1;
    font-size: 12px;
}

.footer {
    color: #64748b;
    font-size: 12px;
    margin-top: 50px;
    border-top: 1px solid #263247;
    padding-top: 20px;
}

@media (max-width: 1000px) {
    .metrics {
        grid-template-columns: repeat(2, 1fr);
    }

    .grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 650px) {
    .sidebar {
        position: relative;
        width: 100%;
        height: auto;
    }

    .main {
        margin-left: 0;
        padding: 20px;
    }

    .metrics {
        grid-template-columns: 1fr 1fr;
    }

    .topbar {
        align-items: flex-start;
        flex-direction: column;
    }
}

</style>

</head>

<body>

<div class="sidebar">

    <div class="logo">✈ Travel<span>Pulse</span></div>

    <div class="subtitle">
        TRAVEL INVENTORY INTELLIGENCE
    </div>

    <div class="nav-title">WORKSPACE</div>

    {% for name, label in pages.items() %}

        <a href="/?page={{ name }}"
           class="{% if page == name %}active{% endif %}">
            {{ label }}
        </a>

    {% endfor %}

    <br>

    <form action="/run" method="POST">
        <button class="button" type="submit">
            ↻ Run Complete Pipeline
        </button>
    </form>

    <div class="footer">
        TravelPulse<br>
        Hackathon Prototype
    </div>

</div>


<div class="main">

    <div class="topbar">

        <div>
            <h1>{{ heading }}</h1>
            <div class="description">{{ description }}</div>
        </div>

        <div class="status">
            PIPELINE: {{ pipeline_status }}
        </div>

    </div>


    {% if page == "overview" %}

        <div class="metrics">

            {% for metric in metrics %}

                <div class="card">

                    <div class="metric-label">
                        {{ metric.label }}
                    </div>

                    <div class="metric-value">
                        {{ metric.value }}
                    </div>

                </div>

            {% endfor %}

        </div>

        <div class="section">

            <div class="section-title">
                Business Intelligence
            </div>

            <div class="grid">

                <div class="card">
                    <div class="section-title">
                        Demand Forecast
                    </div>
                    <div class="chart">
                        {{ forecast_chart | safe }}
                    </div>
                </div>

                <div class="card">
                    <div class="section-title">
                        Procurement Allocation
                    </div>
                    <div class="chart">
                        {{ procurement_chart | safe }}
                    </div>
                </div>

            </div>

        </div>

        <div class="section">

            <div class="section-title">
                Data Availability
            </div>

            <div class="card">

                {% for item in data_status %}

                    <p>{{ item | safe }}</p>

                {% endfor %}

                <p class="description">
                    Last successful run: {{ last_run or "Not available" }}
                </p>

            </div>

        </div>


    {% elif page == "demand" %}

        <div class="section-title">
            Demand Forecast vs Actual Demand
        </div>

        <div class="chart">
            {{ forecast_chart | safe }}
        </div>

        <div class="section">
            <div class="section-title">Forecast Records</div>
            <div class="table-wrap">
                {{ forecast_table | safe }}
            </div>
        </div>


    {% elif page == "opportunities" %}

        <div class="section-title">
            Opportunity Distribution
        </div>

        <div class="chart">
            {{ opportunity_chart | safe }}
        </div>

        <div class="section">
            <div class="section-title">Opportunity Records</div>
            <div class="table-wrap">
                {{ opportunity_table | safe }}
            </div>
        </div>


    {% elif page == "optimizer" %}

        <div class="metrics">

            {% for metric in metrics %}

                <div class="card">
                    <div class="metric-label">{{ metric.label }}</div>
                    <div class="metric-value">{{ metric.value }}</div>
                </div>

            {% endfor %}

        </div>

        <div class="section-title">
            Procurement Allocation
        </div>

        <div class="chart">
            {{ procurement_chart | safe }}
        </div>

        <div class="section">
            <div class="section-title">Optimal Procurement Plan</div>
            <div class="table-wrap">
                {{ optimizer_table | safe }}
            </div>
        </div>


    {% elif page == "scenarios" %}

        <div class="section-title">
            Profit Under Different Demand Conditions
        </div>

        <div class="chart">
            {{ scenario_chart | safe }}
        </div>

        {% if expected_profit is not none %}
            <div class="card section">
                <div class="metric-label">
                    Probability-Weighted Expected Profit
                </div>
                <div class="metric-value">
                    {{ expected_profit }}
                </div>
            </div>
        {% endif %}

        <div class="section">
            <div class="section-title">Scenario Results</div>
            <div class="table-wrap">
                {{ scenario_table | safe }}
            </div>
        </div>

    {% endif %}


    {% if pipeline_logs %}

        <div class="section">

            <div class="section-title">
                Pipeline Execution Logs
            </div>

            {% for log in pipeline_logs %}

                <div class="log">

                    <strong>{{ log.name }}</strong>

                    <span class="{{ 'success' if log.status == 'SUCCESS' else 'failed' }}">
                        {{ log.status }}
                    </span>

                    {% if log.output %}
                        <details>
                            <summary>View execution output</summary>
                            <pre>{{ log.output }}</pre>
                        </details>
                    {% endif %}

                </div>

            {% endfor %}

        </div>

    {% endif %}

</div>

</body>
</html>
"""


# ============================================================
# 7. DASHBOARD ROUTE
# ============================================================

@app.route("/")
def dashboard():

    page = request.args.get("page", "overview")

    pages = {
        "overview": "Executive Overview",
        "demand": "Demand Intelligence",
        "opportunities": "Opportunity Detection",
        "optimizer": "Inventory Optimization",
        "scenarios": "Scenario Analysis"
    }

    if page not in pages:
        page = "overview"

    data = load_data()

    forecast = data["forecast"]
    opportunities = data["opportunities"]
    optimizer = data["optimizer"]
    scenarios = data["scenarios"]

    # -------------------------
    # Forecast chart
    # -------------------------

    forecast_chart = "<p>No forecast data available.</p>"

    if not forecast.empty:

        if "travel_date" in forecast.columns:

            columns = [
                col for col in
                ["predicted_demand", "actual_demand"]
                if col in forecast.columns
            ]

            if columns:

                fig = px.line(
                    forecast,
                    x="travel_date",
                    y=columns,
                    title="Demand Forecast"
                )

                forecast_chart = create_chart(fig)

    # -------------------------
    # Procurement chart
    # -------------------------

    procurement_chart = "<p>No procurement data available.</p>"

    if not optimizer.empty:

        required = {
            "hotel_id",
            "rooms_to_purchase"
        }

        if required.issubset(optimizer.columns):

            fig = px.bar(
                optimizer,
                x="hotel_id",
                y="rooms_to_purchase",
                color=(
                    "supplier_id"
                    if "supplier_id" in optimizer.columns
                    else None
                ),
                title="Procurement Allocation"
            )

            procurement_chart = create_chart(fig)

    # -------------------------
    # Opportunity chart
    # -------------------------

    opportunity_chart = "<p>No opportunity data available.</p>"

    if not opportunities.empty:

        if "opportunity_level" in opportunities.columns:

            fig = px.histogram(
                opportunities,
                x="opportunity_level",
                title="Opportunity Distribution"
            )

            opportunity_chart = create_chart(fig)

    # -------------------------
    # Scenario chart
    # -------------------------

    scenario_chart = "<p>No scenario data available.</p>"

    if not scenarios.empty:

        if {"scenario", "profit"}.issubset(scenarios.columns):

            fig = px.bar(
                scenarios,
                x="scenario",
                y="profit",
                color="scenario",
                title="Scenario Profit Comparison"
            )

            scenario_chart = create_chart(fig)

    # -------------------------
    # Overview metrics
    # -------------------------

    overview_metrics = [
        {
            "label": "Hotels Analyzed",
            "value": f"{get_metric(forecast, 'hotel_id', 'unique'):,}"
        },
        {
            "label": "Predicted Demand",
            "value": f"{get_metric(forecast, 'predicted_demand'):,.0f}"
        },
        {
            "label": "Rooms to Purchase",
            "value": f"{get_metric(optimizer, 'rooms_to_purchase'):,.0f}"
        },
        {
            "label": "Expected Profit",
            "value": f"{get_metric(optimizer, 'expected_profit'):,.2f}"
        }
    ]

    # -------------------------
    # Optimization metrics
    # -------------------------

    optimizer_metrics = [
        {
            "label": "Total Procurement",
            "value": f"{get_metric(optimizer, 'procurement_amount'):,.2f}"
        },
        {
            "label": "Expected Profit",
            "value": f"{get_metric(optimizer, 'expected_profit'):,.2f}"
        },
        {
            "label": "Expected Rooms Sold",
            "value": f"{get_metric(optimizer, 'expected_rooms_sold'):,.0f}"
        }
    ]

    metrics = (
        overview_metrics
        if page == "overview"
        else optimizer_metrics
        if page == "optimizer"
        else []
    )

    # -------------------------
    # Expected scenario profit
    # -------------------------

    expected_profit = None

    if {"profit", "probability"}.issubset(scenarios.columns):

        expected_profit = (
            scenarios["profit"] * scenarios["probability"]
        ).sum()

        expected_profit = f"{expected_profit:,.2f}"

    # -------------------------
    # Data status
    # -------------------------

    data_status = []

    for key, df in data.items():

        label = key.replace("_", " ").title()

        if df.empty:
            data_status.append(
                f"<span class='failed'>● {label}: Missing</span>"
            )
        else:
            data_status.append(
                f"<span class='success'>● {label}: "
                f"{len(df)} records loaded</span>"
            )

    descriptions = {
        "overview": "Monitor demand, inventory and financial performance.",
        "demand": "Understand and predict future travel demand.",
        "opportunities": "Identify hotels with potential inventory opportunities.",
        "optimizer": "Analyze optimal hotel inventory procurement.",
        "scenarios": "Evaluate profit and demand under different conditions."
    }

    return render_template_string(
        HTML,
        page=page,
        pages=pages,
        heading=pages[page],
        description=descriptions[page],
        pipeline_status=pipeline_status,
        pipeline_logs=pipeline_logs,
        last_run=last_run,
        metrics=metrics,
        data_status=data_status,
        forecast_chart=forecast_chart,
        procurement_chart=procurement_chart,
        opportunity_chart=opportunity_chart,
        scenario_chart=scenario_chart,
        forecast_table=create_table(forecast),
        opportunity_table=create_table(opportunities),
        optimizer_table=create_table(optimizer),
        scenario_table=create_table(scenarios),
        expected_profit=expected_profit
    )


# ============================================================
# 8. MANUAL PIPELINE EXECUTION
# ============================================================

@app.route("/run", methods=["POST"])
def run():

    run_pipeline()

    return redirect(url_for("dashboard"))


# ============================================================
# 9. START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\nStarting TravelPulse...")
    print(f"Project directory: {ROOT}")

    # Run every pipeline script once on startup
    run_pipeline()

    print("\nDashboard available at:")
    print("http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
