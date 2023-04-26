import telegram
import datetime

# replace with your own API token
bot_token = 'your_bot_token_here'

# replace with the channel username or ID you want to fetch messages from
channel_username = 'your_channel_username_here'

# set the time period for which you want to fetch messages
start_time = datetime.datetime(2023, 4, 1)
end_time = datetime.datetime(2023, 4, 25)

# create a Telegram bot object using the provided API token
bot = telegram.Bot(token=bot_token)

# get the channel object for the specified username or ID
channel = bot.get_chat(chat_id=channel_username)

# get the channel messages
messages = bot.get_chat_history(chat_id=channel.id)

# filter messages by date and time
filtered_messages = []
for message in messages:
    message_time = datetime.datetime.fromtimestamp(message.date)
    if start_time <= message_time <= end_time:
        filtered_messages.append(message)

# print the filtered messages
for message in filtered_messages:
    print(message.text)