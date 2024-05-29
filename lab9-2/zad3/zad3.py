import praw

reddit = praw.Reddit(client_id='jREMcRdB7ZLw2dROZkn7Mw',
                     client_secret='-m6PyEKDT6O3nqNdKrpTvDrH_2ZmhQ',
                     user_agent='Axelaxdd')

subreddit_name = "worldnews"


subreddit = reddit.subreddit(subreddit_name)
post_titles = [post.title for post in subreddit.new(limit=100)]

with open(f"{subreddit_name}_posts.txt", "w", encoding="utf-8") as file:
    for title in post_titles:
        file.write(title + "\n")

print(f"Pobrano {len(post_titles)} postów z subredditu {subreddit_name} i zapisano do pliku {subreddit_name}_posts.txt")
