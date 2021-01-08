import base64
import datetime
import mimetypes
import os.path
import pickle
from pathlib import Path
from typing import Dict

from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource

def login() -> Resource:
    """
    Authorizes this application with the provided [credentials.json].
    The Gmail API provides a simple way to download this file.
    (Most of this function is copy pasted from their sample code)

    Returns:
        An authorized handle to the gmail API.
    """
    CREDENTIALS_PATH = f'{Path(__file__).parent.absolute()}/credentials.json'
    if not Path(CREDENTIALS_PATH).exists():
        raise FileNotFoundError(f"{CREDENTIALS_PATH} is expected to exist with the credentials downloaded from the Gmail API")

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH,
                ['https://www.googleapis.com/auth/gmail.compose'])
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def send(gmail: Resource, recipients: str, text: str, image_path: Path) -> None:
    """
    Sends an email with text and an image.

    Args:
        gmail      : The handle to the API
        recipients : A string of space-separated emails
        text       : The text to attach in the email
        image_path : The path to the image to attach to the email
    """
    # Send this email in monospace font and using html encoding
    html_text = "<font face='Courier New, Courier, monospace'><pre>" + text + "</pre></font>"
    html_text = html_text.replace("\n", "<br>")
    message = MIMEMultipart()
    message["to"] = recipients
    # @TODO: Move this hardcoded email out
    message["from"] = "jpan127@gmail.com"
    message["subject"] = f"{str(datetime.datetime.now().date())} Covid Metrics"
    message.attach(MIMEText(html_text, "html"))

    # Attach the image
    content_type, _ = mimetypes.guess_type(str(image_path))
    if not content_type or content_type != "image/jpeg":
        raise RuntimeError(f"{image_path} is the wrong file type")
    with image_path.open("rb") as f:
        msg = MIMEImage(f.read(), _sub_type="jpeg")
        msg.add_header("Content-Disposition", "attachment", filename=str(image_path))
    message.attach(msg)

    # @TODO: See if the string can be sent directly
    body: Dict[str, str] = {
        "raw": base64.urlsafe_b64encode(message.as_bytes()).decode()
    }

    # Send the email
    gmail.users().messages().send(userId="jpan127@gmail.com", body=body).execute()
