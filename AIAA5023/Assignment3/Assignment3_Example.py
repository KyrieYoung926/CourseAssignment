from zhipuai import ZhipuAI

times = 0 # The number of times the chatbot has been called
client = ZhipuAI(api_key="")  # Please fill in your own APIKey

def text_generation(prompt, times):
    response = client.chat.completions.create(
        model="glm-4-flash", 
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    times += 1
    return response.choices[0].message, times

############################################################################################################
# The following functions are for you to implement
# You can implement functions for saving and loading files, and planning
# You can also implement functions for the chatbot to interact with the user
############################################################################################################

def savig_files(results):
    # to save to generated_text.txt
    with open("50005050.txt", "w") as f:
        f.write(results)

def load_files():
    pass

def planning():
    pass

def others():
    pass

# Simple example

results, times = text_generation("generating a long text with the topic of: AI and the future of human beings", times)
results, times = text_generation("polish the text，" + results.content, times)

print(results.content)
savig_files(results.content)

############################################################################################################

print("API calls:", times)
