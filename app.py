from flask import Flask, render_template, request, jsonify, redirect, url_for
import os, datetime, sqlite3, json, time, threading
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from transformers import CLIPProcessor, CLIPModel
import torch
import uuid
import re

# IMPORTANT: Ensure Tesseract OCR is installed on your system and its path is configured.
# You can check by running 'tesseract --version' in your terminal.
import pytesseract

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

DB_FILE = 'civic.db'
CONFIDENCE_THRESHOLD = 0.4


# --- Initialize SQLite database with new schema ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS departments
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       name
                       TEXT
                       NOT
                       NULL,
                       email
                       TEXT
                       NOT
                       NULL
                   )
                   ''')
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS issues
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       image
                       TEXT
                       NOT
                       NULL,
                       address
                       TEXT,
                       assigned_dept
                       TEXT,
                       status
                       TEXT,
                       last_update
                       TEXT,
                       citizen_email
                       TEXT,
                       issue_id
                       TEXT
                       UNIQUE,
                       total_days
                       INTEGER,
                       current_day
                       INTEGER
                       DEFAULT
                       0,
                       day_updates
                       TEXT,
                       created_date
                       TEXT
                   )
                   ''')
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM departments")
    if cursor.fetchone()[0] == 0:
        depts = [
            ("Public Works", "joisarvraj@gmail.com"),
            ("Disaster Management", "joisarvraj@gmail.com"),
            ("Municipal Cleaning & Sanitation", "sujithroy0812@gmail.com"),
            ("Electricity Dept", "joisarvraj@gmail.com"),
            ("Water Department", "joisarvraj@gmail.com"),
            ("Transport Department", "joisarvraj@gmail.com")
        ]
        cursor.executemany("INSERT INTO departments (name, email) VALUES (?, ?)", depts)
        conn.commit()
    conn.close()


init_db()

# --- Load CLIP model ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Department prompts ---
DEPT_PROMPT_MAP = {
    "Public Works": "A photo of damaged public infrastructure, such as potholes in a road, cracked sidewalks, a fallen tree blocking a path, a broken bench or a broken chair, or a street flooded from poor drainage.",
    "Disaster Management": "A photo of a major disaster scene, such as severe widespread water flooding, a collapsed building, a landslide covering a road, or a large-scale fire.",
    "Municipal Cleaning & Sanitation": "A photo of a public sanitation issue, such as an overflowing garbage bin, illegally dumped trash, a sewage leak in the street, significant littering, or a dead animal on public property.",
    "Electricity Dept": "A photo of a problem with electrical infrastructure, such as a broken streetlight, a damaged power pole, exposed wires, sparking equipment, or a malfunctioning traffic signal.",
    "Water Department": "A photo of a major water supply problem, such as a burst pipe gushing water, a significant leak from a fire hydrant or water main, or flooding caused by a broken water pipe.",
    "Transport Department": "A photo of damaged public transport infrastructure, such as a shattered bus stop, a vandalized public bus, a broken seat at a station, or a malfunctioning display board, or a broken traffic light.",
    "No Issue": "A photo of a clean, well-maintained public space with no visible problems, damage, or hazards of any kind."
}


# --- Helper Functions ---
def generate_issue_id():
    return str(uuid.uuid4())[:8].upper()


def check_missed_updates():
    """Background thread to check for missed daily updates and send reminders"""
    while True:
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT id, assigned_dept, current_day, total_days, issue_id, last_update
                           FROM issues
                           WHERE status = 'In Progress'
                           """)
            issues = cursor.fetchall()

            for issue in issues:
                issue_id, dept_name, current_day, total_days, issue_uid, last_update_str = issue

                if current_day < total_days:
                    # Check if last update was more than 24 hours ago
                    last_update = datetime.datetime.strptime(last_update_str, "%Y-%m-%d").date()
                    days_since_update = (datetime.date.today() - last_update).days

                    if days_since_update >= 1:
                        # Get department email
                        cursor.execute("SELECT email FROM departments WHERE name = ?", (dept_name,))
                        dept_email = cursor.fetchone()[0]

                        # Send reminder email
                        subject = f"Reminder: Update Required for Issue #{issue_uid}"
                        body = f"""
                        Hello {dept_name} Department,

                        This is a reminder that you need to provide an update for Day {current_day + 1} 
                        for Issue #{issue_uid}. Please log into your portal to provide the update.

                        Thank you,
                        Civic Issue Reporting System
                        """

                        send_email(dept_email, subject, body, None, is_html=False)
                        print(f"Sent reminder to {dept_name} for Issue #{issue_uid}")

            conn.close()
        except Exception as e:
            print(f"Error in reminder thread: {e}")

        # Check every 24 hours
        time.sleep(24 * 60 * 60)


# Start the reminder thread
reminder_thread = threading.Thread(target=check_missed_updates, daemon=True)
reminder_thread.start()


# --- UPDATED FUNCTION WITH MORE ROBUST OCR LOGIC ---
def extract_address_from_text(image_path):
    """
    Attempts to extract an address from text present on the image using OCR.
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        print("Extracted text from image:", text)  # For debugging purposes

        # Use a more flexible regex to find a potential address pattern
        match = re.search(r'([\w\s]+\d*),? ([\w\s]+),? ([\w\s]+),? (India)', text, re.IGNORECASE)
        if match:
            address = match.group(0).strip()
            # Simple check to ensure it's not just a random word
            if len(address.split()) > 3:
                return address

    except pytesseract.TesseractNotFoundError:
        print("Tesseract not installed or not in PATH. Skipping OCR.")
    except Exception as e:
        print(f"OCR or address parsing failed: {e}")

    return None


# --- Department classifier ---
def assign_department_hf_generic(image_path, confidence_threshold=CONFIDENCE_THRESHOLD):
    image = Image.open(image_path).convert("RGB")
    DEPT_LABELS = list(DEPT_PROMPT_MAP.keys())
    DEPT_PROMPTS = list(DEPT_PROMPT_MAP.values())

    inputs = clip_processor(text=DEPT_PROMPTS, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    pred_idx = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item()

    if confidence < confidence_threshold:
        return {"name": "Invalid", "email": None, "confidence": confidence}

    dept_name = DEPT_LABELS[pred_idx]
    if dept_name == "No Issue":
        return {"name": "Invalid", "email": None, "confidence": confidence}

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM departments WHERE name=?", (dept_name,))
    email_row = cursor.fetchone()
    conn.close()
    email = email_row[0] if email_row else None

    return {"name": dept_name, "email": email, "confidence": confidence}


# --- Email ---
def send_email(to_email, subject, body, image_path, is_html=False):
    sender_email = "sujithroy0812@gmail.com"
    sender_password = "cfxedjajjujjmnmk"

    msg = MIMEMultipart("related")
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg_alternative = MIMEMultipart("alternative")
    msg.attach(msg_alternative)

    if is_html:
        # Attach both plain fallback + HTML
        plain_body = "This is the plain-text fallback. Please view in an HTML-enabled client."
        msg_alternative.attach(MIMEText(plain_body, 'plain'))
        msg_alternative.attach(MIMEText(body, 'html'))
    else:
        # Attach the actual plain text body
        msg_alternative.attach(MIMEText(body, 'plain'))

    # Embed the image if available
    if image_path:
        with open(image_path, 'rb') as fp:
            img = MIMEImage(fp.read())
            img.add_header('Content-ID', '<issue_image>')
            msg.attach(img)

    if not to_email:
        print("⚠️ No recipient email provided; skipping email.")
        return

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email sent to {to_email} with subject: {subject}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# --- Routes ---
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/citizen', methods=['GET', 'POST'])
def citizen():
    if request.method == 'POST':
        file = request.files.get('image')
        manual_address = request.form.get('address', '')
        citizen_email = request.form.get('email', '')

        if not file:
            return "No file uploaded"

        # Validate that address is provided
        if not manual_address:
            return "Address is required"

        # Validate that email is provided
        if not citizen_email:
            return "Email is required"

        unique_filename = str(uuid.uuid4()) + "_" + file.filename
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Use the manually provided address
        final_address = manual_address

        best_dept = assign_department_hf_generic(filepath)

        if best_dept['name'] == "Invalid":
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                           INSERT INTO issues (image, address, assigned_dept, status, last_update, citizen_email,
                                               issue_id, created_date)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (unique_filename, final_address, "Invalid", "Rejected",
                                 str(datetime.date.today()), citizen_email, generate_issue_id(),
                                 str(datetime.date.today())))
            conn.commit()
            conn.close()
            return "❌ Not a valid image for civic issue reporting."

        # Generate unique issue ID
        issue_uid = generate_issue_id()

        # Save valid issue
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
                       INSERT INTO issues (image, address, assigned_dept, status, last_update, citizen_email, issue_id,
                                           created_date)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (unique_filename, final_address, best_dept['name'], "Pending",
                             str(datetime.date.today()), citizen_email, issue_uid,
                             str(datetime.date.today())))
        issue_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Send email
        subject = f"[New Civic Report] Issue #{issue_uid} - Assigned to {best_dept['name']}"

        html_body = f"""
        <html>
        <body style='font-family: Arial, sans-serif; line-height: 1.6; color: #333;'>
            <div style='max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;'>
                <h2 style='color: #008080;'>New Civic Issue Reported</h2>
                <hr style='border-color: #ddd;'>
                <p>Hello <b>{best_dept['name']} Department</b>,</p>
                <p>A new civic issue has been submitted and automatically assigned to your department for review and action. Below are the details:</p>

                <ul style='list-style-type: none; padding: 0;'>
                    <li style='margin-bottom: 10px;'><strong>Issue ID:</strong> {issue_uid}</li>
                    <li style='margin-bottom: 10px;'><strong>Assigned Department:</strong> {best_dept['name']}</li>
                    <li style='margin-bottom: 10px;'><strong>Location:</strong> {final_address if final_address else "Address not provided"}</li>
                    <li style='margin-bottom: 10px;'><strong>Date Reported:</strong> {datetime.date.today()}</li>
                </ul>

                <p><strong>Image of the issue:</strong></p>
                <p style='text-align: center;'>
                    <img src="cid:issue_image" alt="Civic Issue Image" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                </p>

                <p>Please log into your portal to view the full details and update the status of this issue. Your prompt attention is appreciated.</p>
                <p>Thank you for your service.</p>

                <hr style='border-color: #ddd;'>
                <p style='font-size: 0.8em; color: #888; text-align: center;'>
                    This is an automated report from the Civic Reporting System.
                </p>
            </div>
        </body>
        </html>
        """
        send_email(best_dept['email'], subject, html_body, filepath, is_html=True)

        return f"✅ Issue submitted successfully! Your Issue ID is: {issue_uid}. Assigned Department: {best_dept['name']}"

    return render_template('citizen.html')


@app.route('/departments')
def departments():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM departments")
    departments = cursor.fetchall()
    conn.close()
    return render_template('departments.html', departments=departments)


@app.route('/department/<int:dept_id>', methods=['GET', 'POST'])
def department_portal(dept_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM departments WHERE id=?", (dept_id,))
    dept_row = cursor.fetchone()
    dept_name = dept_row[0] if dept_row else "Unknown"

    if request.method == 'POST':
        # Handle status updates (existing functionality)
        issue_id = int(request.form.get('issue_id'))
        new_status = request.form.get('status')
        if issue_id and new_status:
            cursor.execute(
                "UPDATE issues SET status=?, last_update=? WHERE id=?",
                (new_status, str(datetime.date.today()), issue_id)
            )
            conn.commit()

    # Get issues for this department
    cursor.execute("""
                   SELECT id,
                          image,
                          address,
                          status,
                          last_update,
                          issue_id,
                          total_days,
                          current_day,
                          created_date
                   FROM issues
                   WHERE assigned_dept = ?
                     AND status != 'Resolved'
                   ORDER BY status, created_date
                   """, (dept_name,))
    issues_data = cursor.fetchall()

    issues = []
    for row in issues_data:
        issue_id, image, address, status, last_update, issue_uid, total_days, current_day, created_date = row

        # Check if days need to be set
        needs_days = status == 'Pending' and total_days is None

        # Check if day update is allowed (24 hours have passed since last update)

        can_update = False
        if status == 'In Progress' and current_day is not None:
            # The button should appear as long as the work is not yet complete.
            can_update = current_day < total_days

        issues.append({
            'id': issue_id,
            'image': image,
            'address': address,
            'status': status,
            'last_update': last_update,
            'issue_uid': issue_uid,
            'total_days': total_days,
            'current_day': current_day,
            'created_date': created_date,
            'needs_days': needs_days,
            'can_update': can_update
        })

    resolved_count = cursor.execute("SELECT COUNT(*) FROM issues WHERE assigned_dept = ? AND status = 'Resolved'",
                                    (dept_name,)).fetchone()[0]
    pending_count = cursor.execute("SELECT COUNT(*) FROM issues WHERE assigned_dept = ? AND status = 'Pending'",
                                   (dept_name,)).fetchone()[0]
    in_progress_count = cursor.execute("SELECT COUNT(*) FROM issues WHERE assigned_dept = ? AND status = 'In Progress'",
                                       (dept_name,)).fetchone()[0]

    conn.close()

    return render_template(
        'dept.html',
        dept_id=dept_id,
        dept_name=dept_name,
        issues=issues,
        resolved_count=resolved_count,
        pending_count=pending_count,
        in_progress_count=in_progress_count
    )


@app.route('/department/<int:dept_id>/set_days/<int:issue_id>', methods=['POST'])
def set_estimated_days(dept_id, issue_id):
    total_days = request.form.get('total_days')

    if not total_days or not total_days.isdigit() or int(total_days) <= 0:
        return "Invalid number of days", 400

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Update the issue with total days, change status to In Progress, and initialize day_updates as empty JSON
    cursor.execute('''
                   UPDATE issues
                   SET total_days  = ?,
                       status      = 'In Progress',
                       last_update = ?,
                       day_updates = '{}'
                   WHERE id = ?
                   ''', (int(total_days), str(datetime.date.today()), issue_id))
    conn.commit()
    conn.close()

    return redirect(url_for('department_portal', dept_id=dept_id))


@app.route('/department/<int:dept_id>/update_day/<int:issue_id>', methods=['POST'])
def update_day_progress(dept_id, issue_id):
    day_number = request.form.get('day_number')
    description = request.form.get('description')

    if not day_number or not description:
        return "Day number and description are required", 400

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Get current issue details
    cursor.execute('''
                   SELECT current_day, total_days, day_updates
                   FROM issues
                   WHERE id = ?
                   ''', (issue_id,))
    issue = cursor.fetchone()

    if not issue:
        conn.close()
        return "Issue not found", 404

    current_day, total_days, day_updates_json = issue

    # Check if this is the next day in sequence
    if int(day_number) != current_day + 1:
        conn.close()
        return "You must update days in sequential order. Please update Day {} first.".format(current_day + 1), 400

    # Parse existing day updates or create new dict
    if day_updates_json:
        day_updates = json.loads(day_updates_json)
    else:
        day_updates = {}

    # Add the new day update
    day_updates[f"Day {day_number}"] = {
        "date": str(datetime.date.today()),
        "description": description
    }

    # Update the issue
    cursor.execute('''
                   UPDATE issues
                   SET current_day = ?,
                       day_updates = ?,
                       last_update = ?
                   WHERE id = ?
                   ''', (int(day_number), json.dumps(day_updates),
                         str(datetime.date.today()), issue_id))

    # Check if this is the last day
    if int(day_number) == total_days:
        # Redirect to resolution page
        conn.commit()
        conn.close()
        return redirect(url_for('resolve_issue', dept_id=dept_id, issue_id=issue_id))

    conn.commit()
    conn.close()

    return redirect(url_for('department_portal', dept_id=dept_id))


@app.route('/department/<int:dept_id>/resolve/<int:issue_id>', methods=['GET', 'POST'])
def resolve_issue(dept_id, issue_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if request.method == 'POST':
        resolution = request.form.get('resolution')

        if resolution == 'resolved':
            # Mark as resolved
            cursor.execute('''
                           UPDATE issues
                           SET status      = 'Resolved',
                               last_update = ?
                           WHERE id = ?
                           ''', (str(datetime.date.today()), issue_id))
        elif resolution == 'extend':
            extra_days = request.form.get('extra_days')

            if not extra_days or not extra_days.isdigit() or int(extra_days) <= 0:
                return "Invalid number of extra days", 400

            # Update total days
            cursor.execute('''
                           SELECT total_days
                           FROM issues
                           WHERE id = ?
                           ''', (issue_id,))
            current_total_days = cursor.fetchone()[0]
            new_total_days = current_total_days + int(extra_days)

            cursor.execute('''
                           UPDATE issues
                           SET total_days  = ?,
                               last_update = ?
                           WHERE id = ?
                           ''', (new_total_days, str(datetime.date.today()), issue_id))

            # Send apology email to citizen
            cursor.execute('''
                           SELECT citizen_email, issue_id
                           FROM issues
                           WHERE id = ?
                           ''', (issue_id,))
            citizen_email, issue_uid = cursor.fetchone()

            subject = f"Update on Your Reported Issue #{issue_uid}"
            body = f"""
            Dear Citizen,

            We apologize for the delay in resolving your reported issue #{issue_uid}. 
            Our department requires additional time to complete the work.

            We estimate {extra_days} additional days will be needed to fully resolve this issue.
            We appreciate your patience and understanding.

            Thank you,
            Civic Issue Reporting System
            """

            send_email(citizen_email, subject, body, None, is_html=False)

        conn.commit()
        conn.close()
        return redirect(url_for('department_portal', dept_id=dept_id))

    # GET request - show resolution form
    cursor.execute('''
                   SELECT issue_id
                   FROM issues
                   WHERE id = ?
                   ''', (issue_id,))
    issue_uid = cursor.fetchone()[0]
    conn.close()

    return render_template('resolve_issue.html', dept_id=dept_id, issue_id=issue_id, issue_uid=issue_uid)


@app.route('/track', methods=['GET', 'POST'])
def track_issue():
    if request.method == 'POST':
        issue_id = request.form.get('issue_id')

        if not issue_id:
            return render_template('track.html', error="Issue ID is required")

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
                       SELECT image,
                              address,
                              assigned_dept,
                              status,
                              created_date,
                              total_days,
                              current_day,
                              day_updates
                       FROM issues
                       WHERE issue_id = ?
                       ''', (issue_id,))
        issue = cursor.fetchone()
        conn.close()

        if not issue:
            return render_template('track.html', error="Issue not found. Please check your Issue ID.")

        # Parse day updates
        day_updates = json.loads(issue[7]) if issue[7] else {}

        return render_template('track_result.html',
                               issue_id=issue_id,
                               image=issue[0],
                               address=issue[1],
                               department=issue[2],
                               status=issue[3],
                               created_date=issue[4],
                               total_days=issue[5] if issue[5] else 0,
                               current_day=issue[6] if issue[6] else 0,
                               day_updates=day_updates)

    # Handle GET request
    error = request.args.get('error')
    return render_template('track.html', error=error)


@app.route('/debug/issue/<issue_id>')
def debug_issue(issue_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
                   SELECT id,
                          issue_id,
                          assigned_dept,
                          status,
                          total_days,
                          current_day,
                          day_updates,
                          last_update
                   FROM issues
                   WHERE issue_id = ?
                   ''', (issue_id,))
    issue = cursor.fetchone()
    conn.close()

    if not issue:
        return "Issue not found"

    result = {
        'id': issue[0],
        'issue_id': issue[1],
        'assigned_dept': issue[2],
        'status': issue[3],
        'total_days': issue[4],
        'current_day': issue[5],
        'day_updates': json.loads(issue[6]) if issue[6] else {},
        'last_update': issue[7]
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)