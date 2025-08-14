from crewai.tools import BaseTool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import uuid

class EDATool(BaseTool):
    name: str = "EDATool"
    description: str = (
        "Performs Exploratory Data Analysis (EDA) on a given CSV file. "
        "Generates histograms and boxplots for numerical features, bar charts for categorical features, "
        "and a correlation heatmap. Input must be a JSON string with 'csv_path'."
    )

    def _run(self, data: str) -> str:
        if not data:
            return "No input provided. Must provide a JSON string with 'csv_path'."

        try:
            parsed = json.loads(data)
            csv_path = parsed.get("csv_path")
            output_dir = parsed.get("output_dir", f"output/eda_charts/eda_charts_{uuid.uuid4().hex[:5]}")

            if not csv_path or not os.path.isfile(csv_path):
                return "Invalid or missing 'csv_path'. Please provide a valid CSV file path."

            return self._perform_eda(csv_path, output_dir)

        except json.JSONDecodeError:
            return "Invalid JSON format. Expecting: {'csv_path': 'path/to/file.csv'}"

    def _perform_eda(self, csv_path: str, output_dir: str) -> str:
        df = pd.read_csv(csv_path)
        os.makedirs(output_dir, exist_ok=True)

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        chart_count = 0

        # 1. Univariate - Numerical
        for col in num_cols:
            # Histogram
            plt.figure()
            df[col].hist(bins=30)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_hist.png")
            plt.close()
            chart_count += 1

            # Boxplot
            plt.figure()
            df.boxplot(column=col)
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_boxplot.png")
            plt.close()
            chart_count += 1

        # 2. Univariate - Categorical
        for col in cat_cols:
            plt.figure()
            df[col].value_counts().plot(kind="bar")
            plt.title(f"Bar Chart of {col}")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_bar.png")
            plt.close()
            chart_count += 1

        # 3. Correlation Heatmap (Fixed)
        if len(num_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_heatmap.png")
            plt.close()
            chart_count += 1

        return json.dumps({
            "status": "success",
            "output_dir": output_dir,
            "charts_generated": chart_count
        })
