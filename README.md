# Campaign Guardian

Campaign Guardian is a Streamlit web application that provides an advanced Ad Campaign Anomaly Detection System. It helps you monitor your ad campaigns, detect anomalies in key metrics, and analyze their impact.

## Features

-   **Anomaly Detection**: Detects anomalies in your ad campaign data using statistical methods.
-   **Interactive Dashboard**: Visualize your campaign data and anomalies in an interactive dashboard.
-   **Campaign and Metric Selection**: Select specific campaigns and metrics to monitor.
-   **Customizable Sensitivity**: Adjust the sensitivity of the anomaly detection algorithm.
-   **Data Export**: Export anomaly reports and campaign summaries to CSV and PDF.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/CampaignGuardian.git
    cd CampaignGuardian
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you don't have a `requirements.txt` file, you can create one from the `pyproject.toml` file or install the dependencies directly.*

## Usage

To run the application, use the following command:

```bash
streamlit run app.py --server.port 8000
```

This will start the Streamlit server and open the application in your web browser.

## Dependencies

The application uses the following Python libraries:

-   `streamlit`
-   `pandas`
-   `plotly`
-   `numpy`
-   `scipy`
-   `kaleido`
-   `reportlab`

You can find the exact versions in the `pyproject.toml` file.
