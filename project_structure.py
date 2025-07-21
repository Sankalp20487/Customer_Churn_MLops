import os

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "modeling",
    "streamlit_app",
    "dashboards/grafana_screenshots",
    "dashboards/powerbi",
    "monitoring",
    "deployment"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created: {folder}")

# Create empty files
files = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "streamlit_app/app.py",
    "streamlit_app/model_loader.py"
]

for file in files:
    with open(file, "w") as f:
        pass
    print(f"📝 Created: {file}")
