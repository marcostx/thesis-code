import os
import slack

def send_msg(string_message):
    token = "xoxp-183901988241-226969512416-1068142375409-f0aa0ab88a40efe38ceeff6ecf607f98"
    channel = "marcos-experiments"

    client = slack.WebClient(token=slack_token)

    response = client.chat_postMessage(
        channel=slack_channel,
        text=string_message)
