from crewai.tools import BaseTool
import matplotlib.pyplot as plt
import os
import json
import uuid


class PlotTool(BaseTool):
    name: str = "PlotTool"
    description: str = (
        "Generates a bar chart from dynamic data retrieved from the given CSV file. "
        "Input must be a JSON string with 'labels', 'values', and optionally 'title', 'xlabel', 'ylabel'."
        "Should be accurate with the user query."
    )

    def _run(self, data: str) -> str:
        if not data:
            return "No data provided. Provide a JSON string with 'labels' and 'values'."

        try:
            parsed = json.loads(data)
            labels = parsed.get("labels")
            values = parsed.get("values")
            title = parsed.get("title", "Bar Chart Visualization")
            xlabel = parsed.get("xlabel", "Category")
            ylabel = parsed.get("ylabel", "Value")

            if not labels or not values or len(labels) != len(values):
                return (
                    "Invalid input. 'labels' and 'values' must be equal-length lists."
                )

            plt.figure(figsize=(8, 5))
            plt.bar(labels, values, color="skyblue")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()

            os.makedirs("output/plot", exist_ok=True)
            plot_path = os.path.join("output/plot", f"plot_{uuid.uuid4().hex[:6]}.png")
            plt.savefig(plot_path)
            plt.close()

            return f"Plot generated: {plot_path}"

        except json.JSONDecodeError:
            return 'Invalid JSON format. Expected: {"labels": [...], "values": [...]}'
