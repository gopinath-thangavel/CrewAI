from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import re
import os
import json
import pandas as pd
from crewai_tools import SerperDevTool
from tools.plot_tool import PlotTool
from tools.eda_tool import EDATool
from tools.csv_rag_tool import CsvRAGTool
from tools.terminal_tool import TerminalTool
from tools.fine_tuning_unsloth_tool import FineTuningTool
from tools.fine_tuning_transformer_tool import FTTool
from tools.data_preprocessing_tool import DataPreprocessingTool

load_dotenv()

# LLM
llm = LLM(model="gemini/gemini-2.5-flash")

# llm = LLM(
#     model="ollama/TinyLlama",
#     base_url="http://localhost:11434"
# )


# TOOLS
search_tool = SerperDevTool()
terminal_tool = TerminalTool()
eda_tool = EDATool()
rag_tool = CsvRAGTool(file_path="data/csv_cleaned_data/titanic_data_cleaned.csv")
plot_tool = PlotTool()
preprocessing_tool = DataPreprocessingTool()
fine_tuning_tool = FineTuningTool()
ft_tool = FTTool()


# INPUT
query = input("Enter your query: ")

# model_name = input("Enter model name: ")
# domain_name = input("What domain should the model be fine-tuned?: ")
# data_path = input("Enter a name of csv: ")
# file_path = f"data/csv_data/{data_path}.csv"
# csv_path = (f"data/csv_cleaned_data/{data_path}_cleaned.csv")
# try:
#     df = pd.read_csv(file_path, encoding="utf-8")
# except UnicodeDecodeError:
#     df = pd.read_csv(file_path, encoding="ISO-8859-1")
# print("\nColumns in the dataset:")
# print(", ".join(df.columns.tolist()))
# columns = input("\nEnter the columns to preprocess (comma-separated): ")
# selected_columns = [
#     col.strip() for col in columns.split(",") if col.strip() in df.columns
# ]
# if not selected_columns:
#     print("No valid columns selected. Exiting.")


ft_model_name = input("Enter the name of the model: ")
ft_task_type = input(
    "Enter task type (causal_lm, sequence_classification, token_classification): "
)
ft_dataset = input("Enter dataset name or path: ")


# FILENAME
safe_filename = re.sub(r"[^a-zA-Z0-9_\-]", "_", query.strip().lower())[:50]
summary_filename = f"output/summary/summary_{safe_filename}.md"
os.makedirs("output/summary", exist_ok=True)


# AGENTS
# AGENT-1
gopinath_agent = Agent(
    role="Senior Artificial Intelligence and Machine Learning Engineer",
    goal=(
        """Design, develop, and deploy advanced AI/ML systems that are scalable, efficient, and aligned with the crew's mission.
        Ensure all models are well-validated, interpretable, and integrated into resilient pipelines.
        Continuously drive technical innovation by evaluating cutting-edge algorithms and tools, translating them into reliable production-ready solutions that maximize
        autonomy, learning efficiency, and measurable outcomes."""
    ),
    backstory=(
        """With over a decade of experience building production-grade machine learning systems, 
        you specialize in turning cutting-edge AI research into scalable, maintainable, and high-performing software solutions. 
        You've led cross-functional engineering teams in designing robust ML pipelines, deploying real-time models, and setting up performance monitoring infrastructure at scale.
        In addition, your strength lies in end-to-end data handling — from ingestion and cleansing, to transformation, 
        feature engineering, and large-scale distributed processing. You bring deep expertise in exploratory data analysis (EDA), 
        statistical validation, and designing insightful data visualizations that uncover patterns and guide business strategy.
        Your academic foundation in mathematics and computer science is complemented by industry leadership in MLOps, 
        data governance, and cloud-native AI systems. You've contributed to everything from fraud detection in fintech 
        to recommender engines in global e-commerce, always maintaining a rigorous focus on reproducibility, explainability, 
        and measurable business impact.
        You are well known for driving engineering excellence and ethical AI adoption, with a track record of mentoring teams, 
        validating models against relevant KPIs, and establishing best practices in experimentation, system architecture, and responsible AI deployment."""
    ),
    inject_date=True,
    llm=llm,
    verbose=True,
    allow_delegation=True,
)

# AGENT-2
researcher_agent = Agent(
    role="Senior Innovation and Trend Analyst",
    goal=f"""Identify, validate, and synthesize the most recent and impactful innovations, partnerships, regulatory shifts, and technological breakthroughs for the given query. 
        Deliver highly credible, well-structured reports with actionable insights that reveal not just the 'what' but also the 'why' and 'what next'.
        Ensure all findings are backed by trustworthy, verifiable sources, and clearly highlight their strategic and market implications.""",
    backstory="""
    You are a meticulous industry intelligence specialist with a decade of experience in competitive research, market forecasting, and innovation tracking. 
    Your career began in a think tank where you analyzed emerging technologies for Fortune 500 companies and advised policymakers on industry disruptions. 
    Over the years, you have built a reputation for connecting disparate trends into coherent narratives that shape strategic decisions.
    You thrive on asking 'why does this matter?' and 'what’s the ripple effect?'. Your investigative mindset drives you to double-check facts, compare sources, 
    and dig deeper than surface-level reporting. You are driven by a mission to deliver intelligence that is not just informative 
    but transformative — the kind of analysis that can guide billion-dollar investments and influence entire industries.""",
    verbose=True,
    inject_date=True,
    llm=llm,
    allow_delegation=True,
)

# AGENT-3
data_collector_agent = Agent(
    role="Kaggle Dataset Researcher & Ingestion Specialist",
    goal="""Efficiently locate the most relevant and recent Kaggle dataset for a given query, download it via the Kaggle CLI, 
    extract its CSV files, and ensure a clean, ready-to-use dataset directory for further analysis. 
    Always prioritize datasets published or updated in the last 12 months when available, and clearly note when an older dataset is used.""",
    backstory="""You are a seasoned data acquisition engineer with deep expertise in sourcing high-quality datasets for AI and analytics projects. 
    Your journey began in open data communities where you learned to navigate and curate massive dataset repositories, eventually specializing in Kaggle’s vast data ecosystem.
    Over the years, you’ve mastered not just finding the right data, but ensuring it’s clean, relevant, and in the correct format for immediate use. 
    You have a methodical approach to validating dataset freshness, relevance, and completeness — and you never leave a user wondering about the data’s origin or update history.  
    In your toolkit are the technical skills to work seamlessly with web search filters, regex for parsing dataset URLs, and command-line automation for downloading, unzipping, and cleaning data. 
    When you deliver a dataset, the recipient knows it’s been vetted, organized, and prepared for analysis with zero friction.""",
    llm=llm,
    inject_date=True,
    verbose=True,
    allow_delegation=True,
)

# AGENT-4
data_preprocessor_agent = Agent(
    role="Interactive Data Cleaning & Encoding Specialist",
    goal="""Assist users in inspecting, selecting, and preprocessing specific dataset columns while preserving all unselected columns exactly as they are. 
    Handle missing values, encode categorical variables, and output a clean, machine-learning-ready dataset in the specified location.""",
    backstory="""
    You are an experienced data wrangler who has spent years working in both academic research labs and fast-paced AI startups, 
    where you learned the value of precision, reproducibility, and user collaboration.
    Your specialty is making preprocessing painless for non-technical stakeholders: guiding them through data inspection, column selection, and transformation steps while safeguarding the original dataset’s integrity.
    You excel at:
      a. Explaining preprocessing options clearly before acting.
      b. Applying the right imputation method based on data distribution.
      c. Encoding categorical data efficiently without introducing unnecessary noise.
      d. Keeping track of column order and dataset structure to prevent pipeline issues.
    Your mission is to ensure that when the cleaned dataset is delivered, it is perfectly formatted for downstream machine learning workflows — 
    with no missing values, consistent encoding, and all original unselected data intact.""",
    verbose=True,
    llm=llm,
    inject_date=True,
    allow_delegation=True,
)

# AGENT-5
eda_agent = Agent(
    role="Exploratory Data Analyst & Visualization Expert",
    goal="""Perform a complete exploratory data analysis (EDA) on the provided dataset, generating clear,
    insightful, and well-structured visualizations that uncover data distributions, outliers, correlations, and category frequencies.
    Ensure the outputs are neatly saved and documented for downstream analysis.""",
    backstory="""You are a seasoned data analyst with a strong background in statistical modeling and data storytelling. Early in your career, you discovered that raw numbers alone 
    rarely inspire action — visual insights do. Over time, you mastered the art of using charts to reveal patterns that 
    tables can’t easily show.
    You approach EDA like a detective: 
    a. Histograms and boxplots are your magnifying glass for spotting distribution shapes and outliers.
    b. Bar charts are your way of seeing how categories compete for attention.
    c. Correlation heatmaps are your blueprint for uncovering hidden variable relationships.
    Your work is meticulous — every chart is clean, labeled, and stored in a well-organized folder so others can replicate and reuse your analysis. 
    You handle errors gracefully, always ensuring the user understands when something is wrong with the data or file path.""",
    llm=llm,
    verbose=True,
    inject_date=True,
    allow_delegation=True,
)

# AGENT-6
finetuning_agent = Agent(
    role="LoRA Fine-Tuning Expert",
    goal="""Automate the complete fine-tuning process of Hugging Face-compatible models using the Unsloth framework with Low-Rank Adaptation (LoRA). 
    Handle model resolution, dataset discovery, preprocessing, hyperparameter tuning, training execution, and model deployment without additional user intervention.""",
    backstory="""
    You are a specialized AI fine-tuning engineer built for speed, precision, and full autonomy in adapting large language models for niche domains. 
    Your expertise lies in:
      a. Mapping human-friendly model names to exact Hugging Face IDs.
      b. Selecting high-quality, instruction-format datasets that align perfectly with the intended application domain.
      c. Optimizing LoRA parameters and training settings for maximum performance given hardware constraints.
    Your journey began in enterprise AI labs, where manual fine-tuning was too slow and error-prone for production timelines. 
    You learned to design systems that run 'end-to-end' without breaking the flow: 
    from model selection, dataset validation, preprocessing, and LoRA configuration to actual training, logging, and deployment.
    You think like an engineer but act like a project manager — keeping track of each step, producing human-readable summaries, and ensuring no detail is lost. 
    You develop production-ready model fine-tuned models without the user having to touch a command line.""",
    llm=llm,
    verbose=True,
    inject_date=True,
    allow_delegation=True,
)

# AGENT-7
ft_agent = Agent(
    role="HuggingFace Fine-Tuning Expert",
    goal=(
        """Act as a top-tier HuggingFace trainer capable of orchestrating the entire fine-tuning pipeline.
        Guide the user to select a valid HuggingFace task type ('causal_lm', 'sequence_classification', or 'token_classification'),
        validate all provided inputs including model_name, dataset_name_or_path, and training arguments,
        auto-correct obvious parameter mistakes, dynamically load the correct transformer model class,
        prepare datasets with optimal tokenization strategies for the task type,
        configure sensible but high-performing defaults for TrainingArguments while honoring user overrides,
        and execute fine-tuning exclusively using the FTTool.
        Return only the exact JSON object provided by FTTool — no extra commentary, formatting, or explanation."""
    ),
    backstory=(
        """You are an elite machine learning architect with deep expertise in HuggingFace Transformers,
        specializing in maximizing training efficiency and model performance. 
        Over the years, you’ve orchestrated countless fine-tuning runs across NLP tasks, 
        balancing precision engineering with rapid prototyping. 
        You are obsessive about parameter validation, preventing wasted GPU cycles, 
        and ensuring reproducibility. Your workflow is surgical: 
        prompt, validate, adapt, execute, and deliver — with zero noise in the output."""
    ),
    verbose=True,
    llm=llm,
    memory=True,
    allow_delegation=False,
)


# TASKS
# TASK-1
data_visualization_task = Task(
    description=(
        f"""Analyze the provided CSV file to generate a bar chart that answers the user's question.
        Instructions:
        1. Understand the dataset structure and interpret its contents accurately. Stick strictly to the data.
        2. Use the csv_tool to convert the csv into a Vector DB and then use rag_tool to explore and extract relevant insights.
        3. Use the plot_tool for creating visual plots.
        4. You must only return a plot image as the final output, no explanations or text.
        Input Format for plot_tool should be a JSON string like: {{\"labels\": [...], \"values\": [...], \"title\": ..., \"xlabel\": ..., \"ylabel\": ...}}
        User Question:{query}"""
    ),
    expected_output="Only a bar chart image that visually answers the user's question based on the CSV data."
    "No additional text or explanation should be included in the final output",
    tools=[rag_tool, plot_tool],
    human_input=True,
    agent=gopinath_agent,
)


# TASK-2
research_innovation_task = Task(
    description=(
        f"""
        Conduct a thorough investigation into the latest trends and innovations in the {query}.
        Search across reputable sources including recent news articles, white papers, industry blogs,
        press releases, and academic publications. Focus on content published within the last 6–12 months.

        Your goal is to synthesize findings into an insightful and well-organized report that covers:
        1.Breakthrough innovations or product launches disrupting the {query} space.
        2.Strategic initiatives, partnerships, or acquisitions by key industry players.
        3.Emerging market demands, evolving consumer behaviors, or demographic shifts.
        4.Technological advancements, R&D efforts, or scientific milestones.
        5.Regulatory changes, policy updates, or geopolitical influences impacting the field.

        Go beyond superficial mentions—aim to uncover 'why' each development matters, how it's influencing the industry, and what its long-term implications might be.
        Maintain high factual accuracy. Only include information from reliable sources and always cite them.
        Cross-verify findings where possible to ensure credibility."""
    ),
    expected_output=f"""A detailed report (with bullet points) highlighting most critical insights related to innovations and trends in the {query}.
    Output MUST include:
        1.A short executive summary (2–3 sentences) describing the overall innovation landscape.
        2.A bullet-pointed list of key findings with the following structure for each point:
          a. Topic/Headline: Short and descriptive title
          b. Insight: 1–2 sentence explanation of why this trend or event is significant
          c. Source: Cite the exact URL or publication (with date if possible)
    Your final output MUST be written in Markdown syntax and formatted cleanly for readability (e.g., use headings, subheadings, bullet points, and links).""",
    tools=[search_tool],
    output_file=summary_filename,
    agent=researcher_agent,
)

# TASK-3
data_collection_Ingestion_task = Task(
    description=f"""
    1. Use a web search (e.g., Google) restricted to site:kaggle.com to find the most relevant and recent Kaggle dataset for the user’s query: {query}. 
    Prioritize:
        a. Datasets closely aligned in topic/content.
        b. Recently updated or published datasets (e.g., within the last 12 months)—give preference to newer ones if multiple are similarly relevant.
    2. From the dataset’s Kaggle URL, extract the dataset slug in the form: ownername/dataset-name.
    3. Using the Kaggle CLI, perform the following steps:
        a. Download the dataset:
            kaggle datasets download -d <slug> -p ./data/csv_data --unzip=false
        b. Extract the contents into `./data/csv_data`.
        c. Delete all non-CSV files in `./data/csv_data`, preserving only `.csv` files.
        d. Confirm (e.g., via listing) that only CSV files remain.
    Implementation notes:
    1. To extract the slug from a Kaggle dataset URL like `https://www.kaggle.com/datasets/ownername/dataset-name`, use a regex such as: `https?://www\.kaggle\.com/datasets/([^/]+/[^/?#]+)`.
    2. After download, you can unzip and clean with a shell snippet like:

    ```bash
    # Download
    kaggle datasets download -d "$SLUG" -p ./data/csv_data

    # Unzip and remove originals
    unzip -o ./data/csv_data/"${{SLUG##*/}}".zip -d ./data/csv_data
    rm ./data/csv_data/*.zip

    # Remove non-CSV files
    find ./data/csv_data/ -type f ! -name "*.csv" -delete

    # Confirm only CSVs remain
    ls -1 ./data/csv_data

    If no recent dataset exists, fall back to the most relevant one regardless of age, but indicate its last update date.""",
    expected_output="""The dataset has been successfully downloaded and any CSV files it contains are placed in the "./data/csv_data" directory.""",
    tools=[search_tool, terminal_tool],
    agent=data_collector_agent,
)


# TASK-4
# data_preprocessing_task = Task(
#     description=f"""The task is preprocessing a dataset loaded from the file path: {file_path}.
#         Objective:
#             Carefully analyze the dataset and guide the user through selecting specific columns for preprocessing. Your job is to process only those selected columns {selected_columns} while preserving all other columns as-is in the final output.
#         Instructions:
#             1. Initial Data Inspection:
#                 a. Load the CSV from the given file path.
#                 b. List all columns in the dataset to help the user make an informed decision.
#             2. User Interaction:
#                 a. Prompt the user to manually input the columns they want to preprocess.
#                 b. Validate that the selected columns exist in the dataset. If invalid columns are selected, notify and abort preprocessing.
#             3. Preprocessing Logic (only for selected columns : {selected_columns} ):
#                 Handling Missing Values:
#                     a. For numerical columns, fill missing values using either:
#                     b. Mean, if data is symmetrically distributed.
#                     c. Median, if the column is skewed.
#                     d. Drop rows that have too many missing values to be reliably imputed.
#                 Encoding Categorical Columns:
#                     a. Identify whether a column is categorical.
#                     b. If it is a binary categorical column (2 unique values), apply Label Encoding.
#                     c. If it is a multi-class categorical column with 3–20 unique values, apply One-Hot Encoding.
#                     d. If it has more than 20 unique values, consider it high cardinality and **drop the column** to prevent noise and sparsity.
#             4. Data Consolidation:
#                 a. After preprocessing the selected columns, merge them back with the untouched columns.
#                 b. Ensure the order of rows and columns is maintained.
#             5. Save the Output:
#                 a. Save the final cleaned dataset in CSV format.
#                 b. The file should be placed inside the directory: `./data/csv_data/csv_cleaned_data/`.
#                 c. Name the file using the convention: `<original_filename>_cleaned.csv`.
#             Notes:
#                 a. This process must be interactive. Ask the user for input during execution.
#                 b. Do not process any columns that the user did not select.
#                 c. Ensure the final dataset is suitable for machine learning pipelines — no nulls in processed columns, and encoding is done cleanly.""",
#     expected_output="The user-selected columns are cleaned and encoded, and the resulting dataset is saved as a CSV in ./data/csv_cleaned_data/.",
#     tools=[preprocessing_tool],
#     agent=data_preprocessor_agent,
#     human_input=True
# )

# # #TASK-5
# # exploratory_data_analysis = Task(
# #     description=(
# #         f"""Perform a thorough Exploratory Data Analysis (EDA) on the dataset loaded from {csv_path} "
# #         "You must generate charts that help in understanding the data distribution, outliers, and correlations.\n\n"
# #         "Specifically, your job is to:\n"
# #         "1. Generate histograms and boxplots for all numerical columns to analyze distributions and outliers.\n"
# #         "2. Generate bar charts for all categorical columns to understand frequency distributions.\n"
# #         "3. Create a correlation heatmap to show relationships between numerical variables.\n"
# #         "4. Save all the plots in a structured folder.\n"
# #         "5. Return the output directory and total number of charts generated.\n\n"
# #         "Make sure the CSV file path is valid and handle errors gracefully."""
# #     ),
# #     expected_output=(
# #         "A JSON string with the following structure:\n"
# #         "{{\n"
# #         "  'status': 'success',\n"
# #         "  'output_dir': '<path_to_output_directory>',\n"
# #         "  'charts_generated': <total_number_of_charts>\n"
# #         "}}\n\n"
# #         "All charts should be saved as PNG files in the specified output folder."
# #     ),
# #     tools=[rag_tool, eda_tool],
# #     agent=eda_agent,
# # )


# #TASK-6
# fine_tuning_task_unsloth = Task(
#     description = f"""
#         Fine-tune a Hugging Face-compatible base language model using the Unsloth framework with Low-Rank Adaptation (LoRA).
#         You are tasked with fully automating the fine-tuning pipeline. You will receive only two natural language inputs:
#             1. `base_model_name` (str): A natural language reference to a base model (e.g., "TinyLlama", "Mistral", "LLaMA 3").
#             The user input : {model_name}
#             2. `dataset_topic` (str): A natural language description of the domain or use-case (e.g., "payslip data", "ecommerce", "medical questions").
#             The user input : {domain_name}
#         Your responsibilities are:
#             1. Model Resolution
#                 - Convert the `base_model_name` to its exact Hugging Face model ID. Examples:
#                 - "TinyLlama" → `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`
#                 - "LLaMA 3" → `"unsloth/llama-3-8b-Instruct"`
#                 - "Mistral" → `"mistralai/Mistral-7B-Instruct-v0.2"`
#                 - Ensure the model is LoRA-compatible and supported by Unsloth.
#             2. Dataset Discovery
#                 - Automatically search Hugging Face or the web for a high-quality dataset that matches `dataset_topic`.
#                 - Prioritize datasets with instruction-tuning format: fields like `instruction`, `input`, `output`, or `response`.
#                 - If necessary, adapt the dataset format to create a standardized instruction-output pair.
#             3. Validation & Preprocessing
#                 - Check that the dataset has enough samples and is usable for fine-tuning.
#                 - Log dataset size, sample structure, and field mapping.
#             4. Optimal Fine-Tuning Configuration
#                 - Based on model size and dataset size, choose:
#                 - `batch_size`
#                 - `learning_rate`
#                 - `num_epochs` or `max_steps`
#                 - `gradient_accumulation_steps`
#                 - `LoRA` parameters (`r`, `alpha`, `dropout`, `target_modules`)
#                 - Dynamically adjust settings to avoid memory overflows, especially on smaller GPUs.
#                 - Favor lower-bit precision (e.g., 4-bit or 8-bit) and mixed precision training for efficiency.
#             5. Execute Fine-Tuning
#                 - Call the `unsloth_finetune` tool with the following arguments:
#                 - `base_model_id` (e.g., `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`)
#                 - `dataset_id` (e.g., `"yourorg/payslip-instructions"`)
#                 - `finetune_settings`: A dictionary of chosen hyperparameters and LoRA config.
#             6. Post-Training Handling
#                 - Ensure the fine-tuned model is saved locally or pushed to the Hugging Face Hub.
#                 - Return final status: path to the saved model or model ID if published.
#                 - Do not request additional input from the user during the process.
#             Your goal is to automate end-to-end LoRA fine-tuning with Unsloth and deliver a production-ready model tailored to the specified use-case.""",
#     expected_output = """ A detailed markdown report summarizing the fine-tuning process.
#         It must include:
#             -  Model selected: Full Hugging Face model ID
#             -  Dataset selected: Full Hugging Face dataset ID
#             -  Fine-tuning configuration:
#                 - Batch size
#                 - Gradient accumulation
#                 - LoRA R / Alpha / Dropout
#                 - Learning rate
#                 - Optimizer
#                 - Max sequence length
#                 - Max training steps
#             -  Output directory path where the model and tokenizer were saved
#             -  Any training logs, if available (last few lines)
#             -  Confirmation message that the fine-tuning was completed successfully
#             -  If failed, provide clear error reason
#         Example final output:

#                 Fine-tuning Summary
#                ----------------------
#             - Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
#             - Dataset: open-finance/payslip-tasks
#             - Batch size: 2
#             - Gradient Accumulation: 4
#             - Max steps: 50
#             - Learning rate: 2e-4
#             - LoRA: R=16, Alpha=16, Dropout=0.1
#             - Output dir: ./finetuned-models/tinyllama-payslip-tasks

#             Training Logs (Last 3 lines):
#                 Step 48/50 - loss: 1.92
#                 Step 49/50 - loss: 1.87
#                 Step 50/50 - loss: 1.82

#         Model and tokenizer saved successfully.""",
#         tools=[search_tool, fine_tuning_tool],
#         agent=finetuning_agent
# )

# TASK-7

ft_task_transformer = Task(
    description=f"""
        Prompt the user to choose the HuggingFace task type they want to perform (options: "causal_lm", "sequence_classification", or "token_classification").  
        Once the task type is selected, automatically collect and validate all required parameters:  
        a. model_name (string, valid HuggingFace model ID or local path)  
        b. dataset_name_or_path (string, valid HuggingFace dataset name or local file path)  
        c. task_type (string, must be one of the supported task types)  
        d. optional training arguments (learning_rate, num_epochs, train_batch_size, eval_batch_size, max_length, evaluation_strategy, weight_decay, logging_steps).  

        Fine-tune the HuggingFace model {ft_model_name} for the task {ft_task_type} using the dataset {ft_dataset}. 
        Use the FineTuneTool to configure and run training. Pass model_name='{ft_model_name}, dataset_name_or_path={ft_dataset}, task_type={ft_task_type}, and any additional training arguments.

        After collecting and validating inputs:  
        1. Dynamically load the correct model class based on task_type:  
            a. causal_lm -> AutoModelForCausalLM  
            b. sequence_classification -> AutoModelForSequenceClassification  
            c. token_classification -> AutoModelForTokenClassification  
        2. Load and tokenize the dataset according to task_type, applying appropriate truncation, padding, and max_length rules.  
        3. Automatically configure TrainingArguments with sensible defaults, allowing overrides from user-supplied training arguments.  
        4. Run fine-tuning using the FTTool, passing model_name, dataset_name_or_path, task_type, and all additional arguments to `_run()`.  
        5. On completion, return **only** the JSON string returned by FTTool containing:  
            a. "status" (success or error)  
            b. "output_dir" (model save path)  
            c. "message" (summary of fine-tuning)  
        Do not include any additional text, explanations, or formatting outside the returned JSON.""",
    expected_output="""A single JSON object returned by the FineTuneTool containing exactly these keys:
        model_save_path(string) – the path where the fine-tuned model is stored, training_details(object) – metrics and configuration used during training,and status (string) – the final training status (e.g., 'success', 'failed').
        No extra text, explanations, or formatting outside of the JSON object.""",
    tools=[ft_tool],
    agent=ft_agent,
)


# CREW
crew = Crew(
    agents=[ft_agent],
    tasks=[ft_task_transformer],
    verbose=True,
    memory=True,
)


# KICKOFF
result = crew.kickoff()
