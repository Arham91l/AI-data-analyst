# 📈 AI Stock Market Analyst

An **AI-powered stock market analysis application** built with **LangChain, Hugging Face, SQL, Pandas, and Streamlit**.

The application combines a traditional quantitative dashboard with an AI analyst that can understand natural-language questions and dynamically select the appropriate analytical tool to answer them.

> **Note:** This project is for educational and analytical purposes only. It does not provide financial or investment advice.

## 🚀 Live Demo

🔗 **[AI Stock Market Analyst](https://ai-data-analyst-5jz3yxfqdlahjlmg8ce558.streamlit.app/)**

---

## ✨ Features

### 📊 Stock Overview

Analyze an individual stock using:

* Latest closing price
* 30-day annualized volatility
* RSI (14)
* 1-year return
* Maximum drawdown
* Candlestick charts
* 20-day SMA
* 50-day SMA
* Trading volume

### 📈 Multi-Stock Comparison

Compare multiple stocks using:

* Normalized price performance
* Daily-return correlation
* Interactive comparison charts
* Correlation heatmap

### 🤖 AI Stock Analyst

Ask questions about the available stocks in natural language.

Examples:

```text
Which ticker had the highest average close?
```

```text
Is META overbought?
```

```text
Compare the volatility of AAPL and TSLA over 60 days.
```

```text
What is the maximum drawdown of NVDA?
```

The AI agent determines which tool is required, executes the analysis, and uses the result to formulate the response.

---

# 🧠 AI Agent Architecture

The project uses a **tool-calling LangChain agent**.

```text
                    User Question
                         │
                         ▼
                 Hugging Face LLM
                         │
                         ▼
                LangChain AI Agent
                         │
              ┌──────────┴──────────┐
              │                     │
        Analytical Tools       SQL Database
              │                     │
     ┌────────┼────────┐            │
     │        │        │            │
    RSI   Volatility  MACD      Aggregations
     │        │        │            │
     └────────┼────────┘            │
              │                     │
              └──────────┬──────────┘
                         ▼
                   Tool Results
                         │
                         ▼
                  Hugging Face LLM
                         │
                         ▼
                   Final Answer
```

The LLM does **not directly calculate market statistics**. Instead, it decides which tool should be used and receives the tool's result before generating the final response.

This helps keep numerical answers grounded in the underlying dataset.

---

# 🛠️ Available Agent Tools

The agent currently has tools for:

| Tool              | Purpose                                |
| ----------------- | -------------------------------------- |
| `list_tables`     | Lists available SQL tables             |
| `get_schema`      | Inspects table columns and sample data |
| `execute_sql`     | Performs read-only SQL queries         |
| `volatility`      | Calculates annualized volatility       |
| `rsi`             | Calculates RSI                         |
| `macd`            | Calculates MACD and signal line        |
| `bollinger_bands` | Calculates Bollinger Bands             |
| `max_drawdown`    | Calculates maximum drawdown            |

### SQL Analytics

SQL is particularly useful for questions involving multiple stocks or aggregation.

For example:

```sql
SELECT Ticker, AVG(Close) AS avg_close
FROM market_data
GROUP BY Ticker
ORDER BY avg_close DESC
LIMIT 1;
```

The agent can use SQL for operations such as:

* `AVG`
* `MAX`
* `MIN`
* `GROUP BY`
* `ORDER BY`
* Filtering
* Ranking
* Cross-ticker aggregation

---

# 📐 Technical Indicators

## Volatility

Annualized volatility is calculated from daily returns:

```text
σannual = σdaily × √252
```

## RSI

RSI is calculated using gains and losses over a configurable window:

```text
RSI = 100 - 100 / (1 + RS)
```

where:

```text
RS = Average Gain / Average Loss
```

## MACD

```text
MACD = EMA(12) - EMA(26)
```

The signal line is:

```text
Signal = EMA(9) of MACD
```

The histogram is:

```text
Histogram = MACD - Signal
```

## Bollinger Bands

```text
Middle Band = SMA(20)

Upper Band = SMA(20) + 2 × σ

Lower Band = SMA(20) - 2 × σ
```

## Maximum Drawdown

```text
Drawdown = Current Price / Running Maximum - 1
```

The minimum value represents the maximum drawdown over the selected period.

---

# 🗄️ Data & SQL

The project uses historical market data stored in:

```text
market_agent_data.csv
```

The dataset contains daily market information including:

* Date
* Ticker
* Open
* High
* Low
* Close
* Volume
* Daily Return
* SMA 20
* SMA 50

The agent creates a SQLite table called:

```text
market_data
```

SQL execution is restricted to **read-only queries** to prevent modification of the underlying data.

---

# 🤗 Hugging Face

The application uses a Hugging Face-hosted model as the LLM backend.

The model can be configured from the Streamlit interface.

The Hugging Face API token is stored securely using Streamlit Secrets and is **not included in the GitHub repository**.

Example:

```toml
HF_TOKEN = "your_hugging_face_token"
```

---

# 🧰 Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Plotly
* Streamlit

### AI / LLM

* LangChain
* LangChain Core
* LangChain Hugging Face
* Hugging Face

### Database

* SQLite
* SQL

### Deployment

* Streamlit Community Cloud
* GitHub

---

# 📁 Project Structure

```text
ai-data-analyst/
│
├── app.py
├── agent.py
├── market_agent_data.csv
├── requirements.txt
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

> `secrets.toml` should remain local and must never be committed to GitHub.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-data-analyst.git
cd ai-data-analyst
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Add your Hugging Face token

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
HF_TOKEN = "hf_your_token"
```

## 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**.

Deployment flow:

```text
GitHub Repository
       │
       ▼
Streamlit Community Cloud
       │
       ├── requirements.txt
       ├── app.py
       ├── agent.py
       └── market_agent_data.csv
              │
              ▼
        Hugging Face API
```

The Hugging Face token should be added through the Streamlit Cloud **Secrets** configuration rather than committed to GitHub.

---

# 🔐 Security

Sensitive credentials are excluded from the repository.

`.gitignore` contains:

```text
.streamlit/secrets.toml
.env
*.db
__pycache__/
```

Never commit API keys, access tokens, passwords, or other credentials to GitHub.

---

# 🎯 Project Goals

This project was built to explore the integration of:

* Large Language Models
* Tool calling
* LangChain agents
* SQL databases
* Quantitative financial analysis
* Natural-language interfaces
* Data visualization
* Streamlit deployment

The main objective is to demonstrate how an LLM can act as an **interface to analytical tools**, rather than simply generating answers from its own knowledge.

---

# 🔮 Future Improvements

Potential future additions include:

* More technical indicators
* Sharpe and Sortino ratios
* Beta and Alpha
* Portfolio analysis
* Advanced SQL analytics
* News-based market analysis
* Financial document RAG
* Real-time market data
* More sophisticated agent workflows
* LangGraph-based multi-step agent architecture
* Improved evaluation and tool-selection monitoring

---

# 👨‍💻 Author

**Arham Ahmed**

AIML Engineering Student | Data Science & AI/ML Enthusiast

Interested in:

* Machine Learning
* Deep Learning
* NLP
* LLMs
* RAG
* AI Agents
* Data Science

---

## ⚠️ Disclaimer

This application is an educational project designed for analyzing historical market data.

It is **not financial advice**, and the outputs should not be interpreted as recommendations to buy, sell, or hold any security.
