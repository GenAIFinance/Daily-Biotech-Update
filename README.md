Biotech Daily Digest
An automated daily biotech news aggregator that collects, categorizes, and delivers personalized biotech industry updates via email. Built with Python and automated through GitHub Actions.
Features

Automated News Collection: RSS feed parsing from major biotech sources
Smart Categorization: Articles automatically sorted into:

Obesity/GLP-1 therapeutics
Cell/Gene therapy
General biotech news
Company-specific updates (ABOS, ALXO, AMGN, REGN, JSPR)


Relevance Scoring: Priority weighting for key therapeutic areas
Professional Email Reports: HTML-formatted daily summaries
Deduplication: Prevents duplicate articles using content hashing
Database Persistence: SQLite storage for article history and analytics
GitHub Actions Automation: Runs daily without manual intervention

Quick Start
Local Setup

Clone the repository:

bashgit clone https://github.com/yourusername/biotech-daily-digest.git
cd biotech-daily-digest

Install dependencies:

bashpip install -r requirements.txt

Create configuration files:

bashcd src
python enhanced_daily_biotech_update.py --create-sources
python enhanced_daily_biotech_update.py --create-env-template

Configure email settings in .env:

envEMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your.email@gmail.com
SMTP_PASSWORD=your-app-password
REPORT_SENDER=your.email@gmail.com
REPORT_RECIPIENTS=recipient@example.com

Test the setup:

bashpython enhanced_daily_biotech_update.py --test-email

Run manually:

bashpython enhanced_daily_biotech_update.py
GitHub Actions Automation

Set up repository secrets in GitHub Settings > Secrets and Variables > Actions:

SMTP_HOST
SMTP_PORT
SMTP_USER
SMTP_PASSWORD
REPORT_SENDER
REPORT_RECIPIENTS


Push to GitHub - the workflow will run automatically at 7 AM UTC daily
Manual trigger available in the Actions tab

Configuration
Email Setup
For Gmail users:

Enable 2-Factor Authentication
Generate an App Password (not your regular password)
Use the App Password in SMTP_PASSWORD

Data Sources
The system monitors RSS feeds from:

Industry Sources: Novo Nordisk, Eli Lilly, BioPharma Dive, FierceBiotech, FiercePharma, Endpoints News, STAT News
Company Sources: Acumen (ABOS), ALX Oncology (ALXO), Amgen (AMGN), Regeneron (REGN), Jasper Therapeutics (JSPR)

Add or modify sources in config/data_sources.json.
Filtering Rules
The system prioritizes articles containing:

High Priority (2x weight): Obesity/GLP-1 keywords (semaglutide, tirzepatide, ozempic, etc.)
Medium Priority (1.5x weight): Cell/Gene therapy keywords (CAR-T, CRISPR, AAV, etc.)
Standard Priority: General oncology and biotech terms

Project Structure
biotech-daily-digest/
├── .github/workflows/          # GitHub Actions automation
├── src/                        # Main application code
├── config/                     # Configuration files
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
Command Line Options
bash# Create sample configuration files
python enhanced_daily_biotech_update.py --create-sources
python enhanced_daily_biotech_update.py --create-env-template

# Test configuration
python enhanced_daily_biotech_update.py --test-email

# Run with custom config
python enhanced_daily_biotech_update.py --config custom_config.yaml
Report Format
Daily reports include:
Industry Update

Obesity/GLP-1: Latest developments in weight management therapeutics
Cell/Gene Therapy: CAR-T, gene editing, and cellular therapeutic news
General: Broader biotech industry updates

Company Updates
Dedicated sections for each tracked company showing recent disclosures and material news.
Scheduling
Default schedule: 7:00 AM UTC daily via GitHub Actions cron job.
To change timing, modify the cron expression in .github/workflows/daily-update.yml:
yamlschedule:
  - cron: '12 * * *'  # 12 PM UTC
Monitoring

Check GitHub Actions tab for execution status
Logs are available as downloadable artifacts
Failed runs can trigger notification workflows

Privacy and Security

Email credentials stored as encrypted GitHub Secrets
No personal data collected or stored
RSS feeds accessed via public endpoints only
Local database contains only article metadata

Troubleshooting
No articles found: Check if RSS feeds are accessible and contain recent content within the lookback window (default: 3 days).
Email not sending: Verify SMTP credentials and ensure App Password is used for Gmail accounts.
GitHub Actions failing: Check repository secrets are properly set and match the required variable names.
Database errors: The SQLite database is created automatically. Delete biotech_updates.db to reset.
Contributing

Fork the repository
Create a feature branch
Make your changes
Test locally
Submit a pull request


Check the GitHub Issues tab
Review GitHub Actions logs
Verify configuration files match the expected format

The system is designed to be maintenance-free once properly configured, providing consistent daily biotech intelligence without manual intervention.
