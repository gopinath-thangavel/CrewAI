from crewai.tools import BaseTool
import subprocess
import os
import zipfile


class TerminalTool(BaseTool):
    name: str = "TerminalTool"
    description: str = "Downloads, unzips Kaggle datasets, and keeps only CSV files."

    def _run(self, command: str) -> str:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                return f"Error:\n{result.stderr}"

            output = f"Command Output:\n{result.stdout}"
            data_dir = "./data/csv_data"
            zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]

            if not zip_files:
                return output + "\nNo ZIP files found in ./data."

            for zip_file in zip_files:
                zip_path = os.path.join(data_dir, zip_file)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                output += f"\nExtracted: {zip_file}"

                for f in os.listdir(data_dir):
                    file_path = os.path.join(data_dir, f)
                    if os.path.isfile(file_path) and not f.endswith(".csv"):
                        os.remove(file_path)
                        output += f"\nDeleted non-CSV: {f}"

            csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            output += (
                "\nFinal CSV files in ./data:\n" + "\n".join(csv_files)
                if csv_files
                else "\nNo CSV files found after cleanup."
            )

            return output

        except Exception as e:
            return f"Exception while running command: {str(e)}"
