# AI Data Scientist Agent 🤖

An intelligent agent-based application for automating data science workflows. This Streamlit application provides a comprehensive solution for data analysis, visualization, cleaning, and modeling with integrated chatbot assistance.

## Features

- **Upload & Schema Analysis**: Upload datasets and automatically analyze schema
- **Data Cleaning**: Interactive data cleaning with AI-powered suggestions
- **Data Visualization**: Create various plots and visualizations
- **Modeling & Evaluation**: Build and evaluate machine learning models
- **Report Generation**: Generate comprehensive analysis reports
- **AI Chatbot**: Get assistance through integrated chatbot functionality

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

## Requirements

See `requirements.txt` for full list of dependencies.

## Project Structure

```
ai-ds-agent/
├── app.py                              # Main application file
├── chatbot.py                          # Chatbot functionality
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker configuration
├── src/
│   └── streamlit_app.py               # Streamlit app components
└── pages/
    ├── 01_📂_Upload_and_Schema.py     # Data upload page
    ├── 02_🧹_Clean_Data.py            # Data cleaning page
    ├── 03_📊_Data_Visualization.py    # Visualization page
    ├── 04_🤖_Modeling_and_Evaluation.py # ML modeling page
    └── 05_📑_Report.py                # Report generation page
```

## License

This project is available for educational and research purposes.
