import sys
import logging
from datetime import datetime
import boto3

# --- basic setup ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ingest")

# --- S3 locations ---
S3_BUCKET = "ece5984-s3-felixt"
RAW_PREFIX = "Genre-Project/raw"
INCOMING_KEY = "Genre-Project/incoming/books.csv"   # source file
DEST_FILENAME = "books.csv"                         # name in raw batch

s3 = boto3.client("s3")


def ingest_genre_data(**context):
    """
    Copies the latest incoming/books.csv file into a timestamped
    folder under raw/, then passes that path to downstream tasks.
    """
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=INCOMING_KEY)
        logger.info(f"Found source file: s3://{S3_BUCKET}/{INCOMING_KEY}")
    except Exception as e:
        logger.error(f"Source file not found: s3://{S3_BUCKET}/{INCOMING_KEY}\n{e}")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    dest_prefix = f"{RAW_PREFIX}/{timestamp}/"
    dest_key = f"{dest_prefix}{DEST_FILENAME}"

    s3.copy_object(
        CopySource={"Bucket": S3_BUCKET, "Key": INCOMING_KEY},
        Bucket=S3_BUCKET,
        Key=dest_key,
    )
    logger.info(f"File copied to: s3://{S3_BUCKET}/{dest_key}")

    try:
        ti = context.get("ti")
        if ti:
            ti.xcom_push(key="raw_prefix", value=dest_prefix)
    except Exception as e:
        logger.warning(f"Could not push XCom value: {e}")