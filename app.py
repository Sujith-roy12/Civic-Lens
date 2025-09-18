from flask import Flask, render_template, request
import os, datetime, sqlite3
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
CONFIDENCE_THRESHOLD = 0.7


# --- Initialize SQLite database ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS departments
                   (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT NOT NULL,
                       email TEXT NOT NULL
                   )
                   ''')
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS issues
                   (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       image TEXT NOT NULL,
                       address TEXT,
                       assigned_dept TEXT,
                       status TEXT,
                       last_update TEXT
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
    "Public Works": "Classify as 'Public Works' only if the image shows clear damage to public infrastructure like potholes, broken roads, cracked sidewalks, fallen trees blocking paths, broken benches, flooded streets, or damaged playground equipment in public parks. If no such problem is visible, do not classify here.",
    "Disaster Management": "Classify as 'Disaster Management' only if the image shows major disaster or emergency damage like severe flooding, storm damage, landslides, collapsed buildings, or large-scale fire damage. For minor issues, choose another department. If no disaster damage is visible, do not classify here.",
    "Municipal Cleaning & Sanitation": "Classify as 'Municipal Cleaning & Sanitation' only if the image shows visible sanitation problems like overflowing garbage bins, illegal dumping, sewage leaks, stagnant water, dead animals on public property, or significant littering. If the area appears clean, do not classify here.",
    "Electricity Dept": "Classify as 'Electricity Dept' only if the image shows electrical issues like broken streetlights, damaged power poles, exposed wires, sparking equipment, or malfunctioning traffic signals. If no electrical problem is visible, do not classify here.",
    "Water Department": "Classify as 'Water Department' only if the image shows water supply problems like burst pipes, major water leaks, flooding from a water main, or vandalized public taps. For minor puddles or rain, choose another department. If no water issue is visible, do not classify here.",
    "Transport Department": "Classify as 'Transport Department' only if the image shows damaged public transport infrastructure like broken bus stops, malfunctioning arrival boards, or damaged public buses. If no transport-related damage is visible, do not classify here.",
    "No Issue": "Classify as 'No Issue' if the image shows normal, well-maintained public spaces with no visible problems."
}


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

    plain_body = f"This is a plain-text version of a civic report. Please view this email in an HTML-enabled client to see the full report and image. Subject: {subject}"
    msg_alternative.attach(MIMEText(plain_body, 'plain'))

    if is_html:
        msg_alternative.attach(MIMEText(body, 'html'))

    # Embed the image
    if image_path:
        with open(image_path, 'rb') as fp:
            img = MIMEImage(fp.read())
            img.add_header('Content-ID', '<issue_image>')  # unique ID for the image
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
        print(f"✅ Email sent to {to_email}")
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

        if not file:
            return "No file uploaded"

        # Validate that address is provided
        if not manual_address:
            return "Address is required"

        unique_filename = str(uuid.uuid4()) + "_" + file.filename
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Use the manually provided address (no longer using geotagging)
        final_address = manual_address

        best_dept = assign_department_hf_generic(filepath)

        if best_dept['name'] == "Invalid":
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                           INSERT INTO issues (image, address, assigned_dept, status, last_update)
                           VALUES (?, ?, ?, ?, ?)
                           ''', (unique_filename, final_address, "Invalid", "Rejected", str(datetime.date.today())))
            conn.commit()
            conn.close()
            return "❌ Not a valid image for civic issue reporting."

        # Save valid issue
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
                       INSERT INTO issues (image, address, assigned_dept, status, last_update)
                       VALUES (?, ?, ?, ?, ?)
                       ''', (unique_filename, final_address, best_dept['name'], "Pending", str(datetime.date.today())))
        issue_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Send email
        subject = f"[New Civic Report] Issue #{issue_id} - Assigned to {best_dept['name']}"

        html_body = f"""
        <html>
        <body style='font-family: Arial, sans-serif; line-height: 1.6; color: #333;'>
            <div style='max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;'>
                <h2 style='color: #008080;'>New Civic Issue Reported</h2>
                <hr style='border-color: #ddd;'>
                <p>Hello <b>{best_dept['name']} Department</b>,</p>
                <p>A new civic issue has been submitted and automatically assigned to your department for review and action. Below are the details:</p>

                <ul style='list-style-type: none; padding: 0;'>
                    <li style='margin-bottom: 10px;'><strong>Issue ID:</strong> {issue_id}</li>
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

        return f"✅ Issue submitted successfully! Assigned Department: {best_dept['name']}"

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
        issue_id = int(request.form.get('issue_id'))
        new_status = request.form.get('status')
        if issue_id and new_status:
            cursor.execute(
                "UPDATE issues SET status=?, last_update=? WHERE id=?",
                (new_status, str(datetime.date.today()), issue_id)
            )
            conn.commit()

    cursor.execute("""
                   SELECT id, image, address, status, last_update
                   FROM issues
                   WHERE assigned_dept = ?
                   ORDER BY id
                   """, (dept_name,))
    issues_data = cursor.fetchall()

    issues = [(i + 1, *row) for i, row in enumerate(issues_data)]

    resolved_count = sum(1 for i in issues if i[4] == 'Resolved')
    pending_count = sum(1 for i in issues if i[4] == 'Pending')

    conn.close()

    return render_template(
        'dept.html',
        dept_name=dept_name,
        issues=issues,
        resolved_count=resolved_count,
        pending_count=pending_count
    )


if __name__ == '__main__':
    app.run(debug=True)