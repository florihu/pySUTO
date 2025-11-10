import smtplib
from email.mime.text import MIMEText

def send_outlook_email(recipient, subject, message):
    sender = "Florian.huber1@wu.ac.at"
    password = "demos25Kratos*"  # or your Outlook password if no 2FA

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP("smtp.office365.com", 587) as server:
            server.starttls()  # Enable TLS
            server.login(sender, password)
            server.send_message(msg)
        print("📩 Email sent via Outlook successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

if __name__ == "__main__":
    send_outlook_email(recipient='Florian96huber@proton.me'
                       , subject='Test Email from Outlook'
                       , message='This is a test email sent from Python using Outlook SMTP server.')