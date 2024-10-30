import datetime
import json
import logging
import mimetypes
import os
import random
import re
import sys
import tempfile
import time
from collections import defaultdict, deque

import ffmpeg
import regex  # For accurate grapheme counting
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from urllib3.util.retry import Retry

load_dotenv()

TWEETS_JS_PATH = "./data/tweets.js.small"
MIGRATED_TWEETS_DIR = "./migrated_tweets/"
FAILED_TWEETS_FILE = "./data/failed_tweets.json"
MEDIA_DIR = "./media/"

USERNAME = os.environ.get("BSKY_USERNAME")
PASSWORD = os.environ.get("PASSWORD")
BASE_URL = os.environ.get("BSKY_BASE_URL", "https://bsky.social")

# Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s", filename="migration.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

# Ensure necessary directories exist
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(MIGRATED_TWEETS_DIR, exist_ok=True)

# Read tweets.js file
with open(TWEETS_JS_PATH, "r", encoding="utf-8") as f:
    content = f.read()
    content = content.replace("window.YTD.tweets.part0 = ", "")
    tweets_data_raw = json.loads(content)

# Build a mapping from tweet ID to tweet data
tweet_id_map = {}
for tweet_entry in tweets_data_raw:
    tweet = tweet_entry["tweet"]
    tweet_id = tweet["id_str"]
    tweet_id_map[tweet_id] = tweet


def create_session_with_retries():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = requests.adapters.HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def login_bsky(identifier, password):
    session = create_session_with_retries()
    url = f"{BASE_URL}/xrpc/com.atproto.server.createSession"
    payload = {"identifier": identifier, "password": password}
    response = session.post(url, json=payload)
    if response.status_code == 200:
        session_data = response.json()
        access_token = session_data["accessJwt"]
        did = session_data["did"]
        handle = session_data["handle"]
        return access_token, did, handle, session
    else:
        logging.error(f"Login failed: {response.status_code} {response.text}")
        return None, None, None, None


def get_text_and_facets(tweet):
    text = tweet["full_text"]
    entities = tweet.get("entities", {})
    facets = []

    # Process URLs
    for url_entity in entities.get("urls", []):
        # Get the indices and URL
        start, end = map(int, url_entity["indices"])
        expanded_url = url_entity["expanded_url"]

        # Replace the URL text in the tweet text with the expanded URL
        text = text[:start] + expanded_url + text[end:]

        # Adjust facets
        byte_start, byte_end = calculate_byte_indices(text, start, start + len(expanded_url))
        facets.append(
            {
                "index": {"byteStart": byte_start, "byteEnd": byte_end},
                "features": [
                    {
                        "$type": "app.bsky.richtext.facet#link",
                        "uri": expanded_url,
                    }
                ],
            }
        )

    # Remove video links from the text
    extended_entities = tweet.get("extended_entities", {})
    for media in extended_entities.get("media", []):
        if media.get("type") == "video":
            url = media.get("url")
            if url:
                text = text.replace(url, "")

    # Process hashtags using the correct facet type
    hashtags = extract_hashtags(text)
    for hashtag_info in hashtags:
        hashtag = hashtag_info["hashtag"][1:]  # Remove '#' from the hashtag
        start = hashtag_info["start"]
        end = hashtag_info["end"]
        byte_start, byte_end = calculate_byte_indices(text, start, end)
        facets.append(
            {
                "index": {"byteStart": byte_start, "byteEnd": byte_end},
                "features": [
                    {
                        "$type": "app.bsky.richtext.facet#tag",
                        "tag": hashtag,
                    }
                ],
            }
        )

    return text, facets


def extract_hashtags(text):
    hashtag_pattern = re.compile(r"#\w+")
    hashtags = []
    for match in hashtag_pattern.finditer(text):
        hashtag = match.group()
        start = match.start()
        end = match.end()
        hashtags.append({"hashtag": hashtag, "start": start, "end": end})
    return hashtags


def calculate_byte_indices(text, char_start, char_end):
    byte_start = len(text[:char_start].encode("utf-8"))
    byte_end = len(text[:char_end].encode("utf-8"))
    return byte_start, byte_end


def truncate_text_and_update_facets(text, facets, max_graphemes=300):
    graphemes = regex.findall(r"\X", text)
    if len(graphemes) <= max_graphemes:
        return text, facets
    else:
        # Determine the truncation point without cutting off any facet
        truncation_point = 0
        current_graphemes = 0
        for i, g in enumerate(graphemes):
            char_index = len("".join(graphemes[:i]))
            # Check if any facet starts after this point
            if current_graphemes >= max_graphemes:
                break
            # Check if this character is within any facet
            within_facet = False
            for facet in facets:
                facet_char_start = len(
                    text.encode("utf-8")[: facet["index"]["byteStart"]].decode("utf-8", errors="ignore")
                )
                facet_char_end = len(text.encode("utf-8")[: facet["index"]["byteEnd"]].decode("utf-8", errors="ignore"))
                if char_index >= facet_char_start and char_index < facet_char_end:
                    within_facet = True
                    break
            if not within_facet:
                current_graphemes += 1
                truncation_point = i + 1  # Since range is exclusive
            else:
                # Include the entire facet
                facet_end_char = facet_char_end
                facet_end_grapheme = len(regex.findall(r"\X", text[:facet_end_char]))
                current_graphemes = facet_end_grapheme
                truncation_point = facet_end_grapheme

        # Truncate text
        truncated_text = "".join(graphemes[:truncation_point]) + "…"

        # Remove facets that are beyond the truncated text
        new_facets = []
        truncated_byte_length = len(truncated_text.encode("utf-8"))
        for facet in facets:
            if facet["index"]["byteEnd"] <= truncated_byte_length:
                new_facets.append(facet)

        return truncated_text, new_facets


def download_media(media_url, tweet_id, media_type):
    try:
        # Make the request to download the media
        response = requests.get(media_url, stream=True)
        response.raise_for_status()

        # Choose extension based on the media type (video or photo)
        if media_type == "video":
            extension = ""  # TODO: fix this
        elif media_type == "photo":
            # Use mimetypes to guess the extension for photos
            extension = mimetypes.guess_extension(response.headers.get("content-type", "").split(";")[0])
            if not extension:
                extension = os.path.splitext(media_url)[1]  # Fallback to extension from URL
        else:
            logging.error(f"Unknown media type {media_type} for media URL {media_url}")
            return None

        # Define the filename
        filename = f"{MEDIA_DIR}{tweet_id}_{os.path.basename(media_url).split('?')[0]}{extension}"

        # Log the media type and file name
        logging.info(f"Downloading {media_type} from {media_url} as {filename}")

        # Download and save the file
        with open(filename, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        return filename
    except Exception as e:
        logging.error(f"Error downloading media {media_url}: {e}")
        return None


def upload_image(filepath, access_token, session):
    url = f"{BASE_URL}/xrpc/com.atproto.repo.uploadBlob"
    mime_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/octet-stream"}
    with open(filepath, "rb") as f:
        data = f.read()
    response = session.post(url, headers=headers, data=data)
    if response.status_code == 200:
        blob_data = response.json()
        blob = blob_data["blob"]
        blob["mimeType"] = mime_type  # Set the correct MIME type
        return blob
    else:
        logging.error(f"Failed to upload image: {response.status_code} {response.text}")
        return None


def upload_video(filepath, access_token, did, session):
    # Get service auth for upload with proper authentication
    service_auth_url = f"{BASE_URL}/xrpc/com.atproto.server.getServiceAuth"
    service_auth_data = {
        "aud": f"did:web:{BASE_URL.split('//')[-1]}",
        "lxm": "com.atproto.repo.uploadBlob",
        "exp": int(time.time()) + 60 * 30,  # 30 minutes expiration
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    # Make a GET request to the service auth URL with proper headers
    service_auth_response = session.get(service_auth_url, params=service_auth_data, headers=headers)
    service_auth = service_auth_response.json()

    if "token" in service_auth:
        token = service_auth["token"]
    else:
        logging.error("Auth token is missing from the service auth response")
        return None

    print("token", token)

    # Upload the video
    file_size = os.path.getsize(filepath)
    upload_url = "https://bsky.social/xrpc/com.atproto.repo.uploadBlob"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "video/mp4",
    }

    with open(filepath, "rb") as file:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading") as pbar:

            def upload_generator():
                while chunk := file.read(4096):
                    pbar.update(len(chunk))
                    yield chunk

            try:
                upload_response = session.post(
                    upload_url,
                    headers=headers,
                    data=upload_generator(),
                )
            except requests.exceptions.SSLError as e:
                logging.error(f"SSL error occurred during video upload: {str(e)}")
                return None

    # Check the upload response
    upload_result = upload_response.json()
    print(upload_result)
    job_id = upload_result["jobId"]
    blob = None
    while not blob:
        job_status_url = "https://video.bsky.app/xrpc/app.bsky.video.getJobStatus"
        job_status_response = session.get(job_status_url, params={"jobId": job_id})
        job_status = job_status_response.json()
        print(f"Status: {job_status}")

        if job_status.get("jobStatus") == "JOB_STATE_FAILED":
            logging.error(f"Video upload failed: {job_status.get('errorMessage')}")
            return None

        blob = job_status.get("blob")
        if not blob:
            time.sleep(1)

    if upload_response.status_code == 200:
        blob_data = upload_response.json()
        blob = blob_data["blob"]
        blob["mimeType"] = "video/mp4"
        return blob
    else:
        logging.error(f"Failed to upload video: {upload_response.status_code} {upload_response.text}")
        return None


def create_post(record, access_token, did, session):
    url = f"{BASE_URL}/xrpc/com.atproto.repo.createRecord"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {"collection": "app.bsky.feed.post", "repo": did, "record": record}
    response = session.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        post_data = response.json()
        return post_data
    else:
        logging.error(f"Failed to create post: {response.status_code} {response.text}")
        return None


def save_migrated_tweet(tweet_id, bluesky_post_info):
    try:
        filename = os.path.join(MIGRATED_TWEETS_DIR, f"{tweet_id}.json")
        with tempfile.NamedTemporaryFile("w", delete=False, dir=MIGRATED_TWEETS_DIR) as tf:
            json.dump(bluesky_post_info, tf)
            temp_name = tf.name
        os.replace(temp_name, filename)
    except Exception as e:
        logging.error(f"Error saving migrated tweet {tweet_id}: {e}")


def is_tweet_migrated(tweet_id):
    filename = os.path.join(MIGRATED_TWEETS_DIR, f"{tweet_id}.json")
    return os.path.exists(filename)


def get_migrated_tweet_info(tweet_id):
    filename = os.path.join(MIGRATED_TWEETS_DIR, f"{tweet_id}.json")
    try:
        with open(filename, "r") as f:
            bluesky_post_info = json.load(f)
        return bluesky_post_info
    except Exception as e:
        logging.error(f"Error reading migrated tweet info for {tweet_id}: {e}")
        return None


def record_failed_tweet(tweet_id, error_message):
    failed_tweet = {"tweet_id": tweet_id, "error": error_message}
    try:
        if os.path.exists(FAILED_TWEETS_FILE):
            with open(FAILED_TWEETS_FILE, "r") as f:
                failed_tweets = json.load(f)
        else:
            failed_tweets = []
        failed_tweets.append(failed_tweet)
        with open(FAILED_TWEETS_FILE, "w") as f:
            json.dump(failed_tweets, f)
    except Exception as e:
        logging.error(f"Error recording failed tweet {tweet_id}: {e}")


def post_to_bluesky(full_text, media_files, reply_to, created_at_iso, access_token, did, facets, session):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            record = {
                "$type": "app.bsky.feed.post",
                "text": full_text,
                "createdAt": created_at_iso,
            }

            # Handle replies
            if reply_to:
                record["reply"] = {
                    "root": {"cid": reply_to["root_cid"], "uri": reply_to["root_uri"]},
                    "parent": {"cid": reply_to["cid"], "uri": reply_to["uri"]},
                }

            # Handle media
            if media_files:
                images = []
                videos = []
                for media_file in media_files:
                    mime_type = mimetypes.guess_type(media_file)[0]
                    if mime_type and mime_type.startswith("image/"):
                        blob = upload_image(media_file, access_token, session)
                        if blob:
                            images.append({"image": blob, "alt": ""})  # Optionally, set alt text
                        else:
                            logging.error(f"Failed to upload media file {media_file}")
                    elif mime_type and mime_type.startswith("video/"):
                        blob = upload_video(media_file, access_token, did, session)
                        if blob:
                            aspect_ratio = get_aspect_ratio(media_file)
                            videos.append({"video": blob, "aspectRatio": aspect_ratio})
                        else:
                            logging.error(f"Failed to upload video file {media_file}")
                    else:
                        logging.error(f"Unsupported media type for file {media_file}")

                if images and not videos:
                    record["embed"] = {"$type": "app.bsky.embed.images", "images": images}
                elif videos and not images:
                    # Only one video is allowed per post
                    record["embed"] = {
                        "$type": "app.bsky.embed.recordWithMedia",
                        "record": {"$type": "app.bsky.embed.record", "record": {}},
                        "media": {"$type": "app.bsky.embed.video", "videos": videos},
                    }
                elif images and videos:
                    # If both images and videos, log and proceed with images
                    logging.error("Cannot upload both images and videos in one post. Proceeding with images.")
                    record["embed"] = {"$type": "app.bsky.embed.images", "images": images}

            # Include facets for hashtags and URLs
            if facets:
                record["facets"] = facets

            # Post the record
            post_data = create_post(record, access_token, did, session)
            if post_data:
                bluesky_post_info = {"uri": post_data["uri"], "cid": post_data["cid"]}
                return bluesky_post_info
            else:
                return None
        except Exception as e:
            wait_time = 2**attempt + random.random()
            logging.error(f"Error posting to Bluesky (attempt {attempt + 1}): {e}")
            logging.info(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    logging.error("Max retries reached. Skipping this tweet.")
    return None


def get_aspect_ratio(file_name: str):
    probe = ffmpeg.probe(file_name)
    video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    if video_stream:
        return {
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"]),
        }
    return None


def cut_video(filename):
    try:
        probe = ffmpeg.probe(filename)
        video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
        if video_stream:
            duration = float(video_stream["duration"])
            if duration > 60:
                logging.info(f"Trimming video {filename} to 60 seconds")
                output_filename = filename.replace(".mp4", "_trimmed.mp4")
                ffmpeg.input(filename, ss=0, t=60).output(output_filename).run(overwrite_output=True)
                os.remove(filename)
                os.rename(output_filename, filename)
    except Exception as e:
        logging.error(f"Error probing video {filename}: {e}")


def process_tweet(tweet_id, access_token, did, processed_tweets, session):
    if is_tweet_migrated(tweet_id):
        logging.info(f"Skipping tweet {tweet_id} (already migrated)")
        return True

    if tweet_id in processed_tweets:
        # Already processed in this run
        return True

    tweet = tweet_id_map.get(tweet_id)
    if not tweet:
        logging.error(f"Tweet {tweet_id} not found in tweet_id_map.")
        return False

    # Check if the tweet is a retweet
    full_text = tweet.get("full_text")
    if not full_text or full_text.startswith("RT @"):
        logging.info(f"Skipping retweet {tweet_id}")
        return True

    processed_tweets.add(tweet_id)

    in_reply_to_status_id = tweet.get("in_reply_to_status_id_str")
    reply_to_info = None
    if in_reply_to_status_id:
        # Ensure parent tweet is processed first
        if not is_tweet_migrated(in_reply_to_status_id):
            parent_processed = process_tweet(in_reply_to_status_id, access_token, did, processed_tweets, session)
            if not parent_processed:
                logging.error(f"Failed to process parent tweet {in_reply_to_status_id} for tweet {tweet_id}")
                return False
        parent_info = get_migrated_tweet_info(in_reply_to_status_id)
        if parent_info:
            # Get root info
            root_uri = parent_info.get("root_uri", parent_info["uri"])
            root_cid = parent_info.get("root_cid", parent_info["cid"])
            reply_to_info = {
                "uri": parent_info["uri"],
                "cid": parent_info["cid"],
                "root_uri": root_uri,
                "root_cid": root_cid,
            }

    logging.info(f"Processing tweet {tweet_id}")

    # Get the tweet text and facets
    full_text, facets = get_text_and_facets(tweet)

    # Truncate text and update facets if it exceeds 300 graphemes
    full_text, facets = truncate_text_and_update_facets(full_text, facets, max_graphemes=300)

    # Get any media associated with the tweet
    media_files = []
    media_entities = []
    if "extended_entities" in tweet:
        media_entities = tweet["extended_entities"].get("media", [])
    elif "entities" in tweet:
        media_entities = tweet["entities"].get("media", [])

    for media in media_entities:
        media_type = media.get("type")
        expanded_url = media.get("expanded_url")
        if media_type == "photo" and "video" in expanded_url:
            continue  # Skip thumbnails

        video_info = media.get("video_info")
        if video_info:
            variants = video_info.get("variants")
            if variants:
                # Get the highest bitrate video
                media_url = max(variants, key=lambda v: int(v.get("bitrate", 0))).get("url")
                if not media_url:
                    continue
                filename = download_media(media_url, tweet_id, media_type)

                # check video duration and cut to 60 seconds if needed
                if filename:
                    cut_video(filename)
        else:
            media_url = media.get("media_url_https") or media.get("media_url")
            if not media_url:
                continue

        if filename:
            media_files.append(filename)

    # Parse tweet creation date
    date_string = tweet.get("created_at")
    date_format = "%a %b %d %H:%M:%S %z %Y"
    parsed_date = datetime.datetime.strptime(date_string, date_format)
    iso_created_at = parsed_date.isoformat()

    # Post to Bluesky
    bluesky_post_info = post_to_bluesky(
        full_text, media_files, reply_to_info, iso_created_at, access_token, did, facets, session
    )
    if bluesky_post_info:
        # Record the mapping
        # If this tweet is a reply, set root_uri and root_cid accordingly
        if reply_to_info:
            root_uri = reply_to_info["root_uri"]
            root_cid = reply_to_info["root_cid"]
        else:
            root_uri = bluesky_post_info["uri"]
            root_cid = bluesky_post_info["cid"]
        bluesky_post_info.update({"root_uri": root_uri, "root_cid": root_cid})
        save_migrated_tweet(tweet_id, bluesky_post_info)
        logging.info(f"Successfully migrated tweet {tweet_id}")
        return True
    else:
        error_message = f"Failed to post tweet {tweet_id} to Bluesky"
        logging.error(error_message)
        record_failed_tweet(tweet_id, error_message)
        return False


def build_dependency_graph():
    graph = defaultdict(set)
    indegree = defaultdict(int)

    for tweet_id, tweet in tweet_id_map.items():
        in_reply_to_status_id = tweet.get("in_reply_to_status_id_str")
        if in_reply_to_status_id:
            graph[in_reply_to_status_id].add(tweet_id)
            indegree[tweet_id] += 1
            if in_reply_to_status_id not in indegree:
                indegree[in_reply_to_status_id] = indegree.get(in_reply_to_status_id, 0)
        else:
            if tweet_id not in indegree:
                indegree[tweet_id] = indegree.get(tweet_id, 0)

    return graph, indegree


def topological_sort(graph, indegree):
    queue = deque()
    order = []

    # Start with nodes that have indegree 0
    for node in indegree:
        if indegree[node] == 0:
            queue.append(node)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(indegree):
        logging.warning("Cycle detected in tweet dependencies. Some replies may not be properly linked.")
        # Return the nodes processed so far
        return order

    return order


def process_tweets(access_token, did, session):
    logging.info(f"{len(tweet_id_map)} tweets to migrate")

    graph, indegree = build_dependency_graph()
    processing_order = topological_sort(graph, indegree)

    processed_tweets = set()

    try:
        for tweet_id in processing_order:
            tweet = tweet_id_map.get(tweet_id)

            full_text = tweet.get("full_text")
            if not full_text or full_text.startswith("RT @"):
                logging.info(f"Skipping retweet {tweet_id}")
                continue

            if is_tweet_migrated(tweet_id):
                logging.info(f"Skipping tweet {tweet_id} (already migrated)")
                continue

            success = process_tweet(tweet_id, access_token, did, processed_tweets, session)
            if not success:
                logging.error(f"Failed to process tweet {tweet_id}")
            time.sleep(0.015)
    except KeyboardInterrupt:
        logging.info("Migration interrupted by user. Exiting...")
        sys.exit(0)


def main():
    access_token, did, handle, session = login_bsky(USERNAME, PASSWORD)
    if not access_token:
        logging.error("Exiting due to login failure.")
        return
    process_tweets(access_token, did, session)


if __name__ == "__main__":
    main()
