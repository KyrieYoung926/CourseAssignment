from zhipuai import ZhipuAI
import time

times = 0
client = ZhipuAI(api_key="6d5a58010e2bf9e564b2aaea8c233627.5IAcpKe37uRdXiBu")

def text_generation(prompt, times):
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        times += 1
        # 添加短暂延迟以避免API限制
        time.sleep(1)
        return response.choices[0].message, times
    except Exception as e:
        print(f"Error in API call: {e}")
        return None, times

def save_files(results):
    with open("50005050.txt", "w", encoding='utf-8') as f:
        f.write(results)

def get_word_count(text):
    words = text.split()
    return len(words)

def generate_section(topic, section_prompt, previous_content=""):
    global times
    context = f"{previous_content}\n\n" if previous_content else ""
    prompt = f"{context}Please write a detailed section about {topic}: {section_prompt}. Make it comprehensive and at least 1500 words."
    results, times = text_generation(prompt, times)
    return results.content if results else ""

def expand_section(section_content, aspect):
    global times
    prompt = f"""Based on this content: {section_content}
    Please expand it significantly by adding more details about {aspect}. 
    Add at least 1500 words of new content focusing on {aspect}, while maintaining coherence with the existing text."""
    results, times = text_generation(prompt, times)
    return results.content if results else ""

def generate_long_text(main_topic):
    global times
    full_text = ""
    
    # 1. 生成主要章节
    sections = [
        ("Introduction", "Provide a comprehensive introduction to the topic, including historical context and current significance"),
        ("Technical Aspects", "Detailed technical analysis and current state of technology"),
        ("Social Impact", "Analysis of social implications and changes"),
        ("Economic Implications", "Economic effects and business transformations"),
        ("Future Predictions", "Future scenarios and potential developments"),
        ("Challenges", "Current and potential future challenges"),
        ("Solutions", "Proposed solutions and recommendations"),
        ("Case Studies", "Real-world examples and applications"),
        ("Ethical Considerations", "Ethical implications and considerations")
    ]
    
    # 2. 为每个章节生成内容
    for section_title, section_prompt in sections:
        print(f"Generating section: {section_title}")
        section_content = generate_section(main_topic, section_prompt, full_text)
        full_text += f"\n\n## {section_title}\n\n{section_content}"
        
        # 3. 为每个章节展开更多细节
        expansion_aspects = [
            "practical applications",
            "recent developments",
            "expert opinions",
            "statistical data and research findings",
            "international perspectives"
        ]
        
        for aspect in expansion_aspects[:2]:  # 只取前两个aspect以控制API调用次数
            expanded_content = expand_section(section_content, aspect)
            full_text += f"\n\n### {aspect.title()}\n\n{expanded_content}"
    
    return full_text

# Main execution
topic = "AI and the future of human beings"
print(f"Starting generation for topic: {topic}")
print("This may take several minutes...")

final_results = generate_long_text(topic)

# 统计信息
word_count = get_word_count(final_results)
print(f"\nTotal words: {word_count}")
print(f"Character count: {len(final_results)}")
print(f"Number of API calls: {times}")

# 保存结果
save_files(final_results)