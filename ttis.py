import pandas as pd
import openai
from openai import OpenAI 
from typing import List, Dict, Union, Optional

import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

def generate_user_input(
        client: openai.OpenAI,
        tags: List[str],
        scaling_value: int,
        title: Optional[str] = None,
        desc: Optional[str] = None,
        sorted_tags: Optional[bool] = None
) -> str:

    print()
    print(tags)
    print(type(tags))
    print()

    # Handle the sorted_tags parameter
    if sorted_tags is False:
        # If the order matters, we keep the tags as is
        tags_formatted = ', '.join(f"'{tag}'" for tag in tags)
        importance_note = "Note: The order of the tags indicates their importance, with earlier tags being more significant."
    else:
        # If the tags are sorted or order doesn't matter, we sort them
        # tags = sorted(tags)
        tags_formatted = ', '.join(f"'{tag}'" for tag in tags)
        importance_note = ""


    # Calculate max tokens
    calculated_max_tokens = int(scaling_value * len(tags))
    print(f"Max tokens = {calculated_max_tokens}")
    
    if calculated_max_tokens >= 700:
        system_prompt = "You are an expert in transforming image tags into vivid and exceptionally detailed image descriptions."
        length_instruction = "Feel free to be very detailed in your description."
    elif calculated_max_tokens >= 500:
        system_prompt = "You are an expert in transforming image tags into vivid and highly detailed image descriptions."
        length_instruction = "Feel free to be detailed in your description."
    elif calculated_max_tokens >= 300:
        system_prompt = "You are an expert in transforming image tags into vivid and moderately detailed image descriptions."
        length_instruction = "Provide a moderately detailed description."
    else:
        system_prompt = "You are an expert in transforming image tags into concise and vivid image descriptions."
        length_instruction = "Please keep the description to no more than two sentences."
    
    # Build the user prompt
    user_prompt = f"""
Given the following list of image tags:

[{tags_formatted}]
"""
    # Include importance note if applicable
    if importance_note:
        user_prompt += f"\n{importance_note}\n"
    
    # Include title and description if provided
    if title:
        user_prompt += f"\nTitle of the image: '{title}'.\n"
    if desc:
        user_prompt += f"\nDescription of the image: '{desc}'.\n"
    
    # Continue the prompt
    user_prompt += f"""
Generate a coherent description of an image that includes the main elements represented by these tags. {length_instruction} The description should be in natural language, suitable as a prompt for an image generation model. Ensure it is fluent, vivid, and captures the essence of the tags.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], 
        max_tokens=calculated_max_tokens,
        temperature=0.7,
        n=1,
        stop=None
    )
    
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Load the API Key
    load_dotenv(".env")

    # Create the tags
    image_tags = input("Please provide the tags that you would like to be converted into a input sequence.  Provide the tags in the following format: \ntag1, tag2, tag3, ..., tagn\n")
    image_tags = image_tags.split(", ")

    # Further Queries
    print()
    print("If Applicable, please provide text, otherwise just press 'enter'")
    title = input("If you have the image title please enter it here: \t")
    desc = input("If you have the image description please enter it here: \t")
    print()

    # Set whether the input is sorted or not
    sorted = input("If the input is in a certain order (By Importance) please type 't', otherwise (no-sort) press enter:\t ")
    if sorted == 't' or 'T' or 'True':
        sorted_tags = True
    else:
        sorted_tags = False

    # This value can be adjusted, higher numbers increase the amount of tokens per input tag
    value = 9.523809523809524

    # Load the OpenAI Client
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

    result = generate_user_input(
        client=client,
        tags=image_tags,
        scaling_value=value,
        title=title,
        desc=desc,
        sorted_tags=sorted_tags
    )
    
    print(result)
