# Flask User Profile App

A modular Flask application with user registration, login, profile editing, file uploads, and secured routes.

## Requirements
- Python 3.11+
- Flask
- Flask-Login
- Flask-WTF
- Flask-SQLAlchemy
- Werkzeug

## Setup
1. Create and activate a virtual environment:

```bash
cd /Users/udaykakani/Projects/Courses/Python_01/Flask/flask
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python run.py
```

4. Open the app in a browser at `http://127.0.0.1:5000/`.

## Notes
- The SQLite database is created automatically at `app.db`.
- Uploaded profile images are saved to `app/static/uploads`.
- The default profile image is served from `app/static/images/default_profile.svg`.
