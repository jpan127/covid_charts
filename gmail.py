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
                f'{Path(__file__).parent.absolute()}/credentials.json',
                ['https://www.googleapis.com/auth/gmail.compose'])
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def send(gmail: Resource, recipients: str, text: str, image_path: Path) -> None:
    html_text = "<font face='Courier New, Courier, monospace'><pre>" + text + "</pre></font>"
    html_text = html_text.replace("\n", "<br>")
    message = MIMEMultipart()
    message["to"] = recipients
    message["from"] = "jpan127@gmail.com"
    message["subject"] = f"{str(datetime.datetime.now())} Covid Metrics"
    message.attach(MIMEText(html_text, "html"))

    content_type, _ = mimetypes.guess_type(str(image_path))
    if not content_type or content_type != "image/jpeg":
        raise RuntimeError(f"{image_path} is the wrong file type")
    with image_path.open("rb") as f:
        msg = MIMEImage(f.read(), _sub_type="jpeg")
        msg.add_header("Content-Disposition", "attachment", filename=str(image_path))
    message.attach(msg)

    body: Dict[str, str] = {
        "raw": base64.urlsafe_b64encode(message.as_string().encode("utf-8")).decode("utf-8")
    }

    gmail.users().messages().send(userId="jpan127@gmail.com", body=body).execute()
