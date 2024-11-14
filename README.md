# twitter-to-bsky importer

Import your Twitter archive into a Bluesky account.

> **⚠ Note:** At this time, this will spam others timelines, so it is only recommended to run on new accounts with no followers.** See `Known problems` below for details.

## Usage

First you'll need your [Twitter archive](https://x.com/settings/download_your_data). You can download it from your Twitter settings. It will be a zip file with a `data` folder inside, containing a `tweet.js` file. The script will look for the `data` folder in the same directory as the `main.py` file.

## Requirements

- Python 3.10+
- ffmpeg (for video conversion / cropping / posting)

## Dependencies

Create a `.env` file (or fill the environment variables in some other way) with the following keys:

```
BSKY_USERNAME=your_username
PASSWORD=your_password
```

> **⚠ Note:** It's reccomended to use [App Passwords](https://blueskyfeeds.com/en/faq-app-password) to avoid using your main password in external applications.

Then run:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Known problems

- Will spam others timelines, so it is only recommended to run on new accounts with no followers.
- Will try to transform hashtags and links into Bsky facets, but may fail.
- If there's a link, will try to use the short form (t.co) to save on characters, but may fail.
- Will try its best to import all tweets, but may fail due to rate limiting, videos or images being too long / too large, etc.
- Longer videos will be cropped to 1 minute.
- Has a very simple retry mechanism, but it's not perfect.
- Will save the ids of migrated tweets to the `migrated_tweets` file, so you can just try again if it fails.
- If it finds a reply, it will try it's best to find the parent tweet and reply to it, but it may fail and just post a new tweet. If the parent tweet is not found (or isn't yours), it will ignore the reply.

## Thanks to

- Boris Mann
- Shinya Kato
- [Heartade](https://github.com/Heartade)

## See also

https://github.com/ianklatzco/atprototools -- underlying library
