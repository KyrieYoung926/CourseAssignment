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

def generate_comprehensive_section(topic, section_info):
    global times
    prompt = f"""Write a comprehensive, in-depth section about {topic}, focusing on {section_info['title']}.

    Required structure and content:
    1. {section_info['main_points'][0]} (at least 1500 words)
    2. {section_info['main_points'][1]} (at least 1500 words)
    3. {section_info['main_points'][2]} (at least 1500 words)

    Requirements:
    - Make each subsection at least 1500 words
    - Include detailed examples and case studies
    - Incorporate relevant statistics and research findings
    - Add expert opinions and different perspectives
    - Ensure smooth transitions between subsections
    - Include practical applications and real-world implications

    Total output should be at least 4500 words for this section."""

    results, times = text_generation(prompt, times)
    return results.content if results else ""

def generate_long_text(main_topic):
    global times
    full_text = ""

    # 重新设计章节结构，每个章节包含更多内容
    sections = [
        {
            "title": "Current State and Future Trajectory",
            "main_points": [
                "Comprehensive analysis of current AI technology and its evolution",
                "Detailed exploration of future predictions and potential developments",
                "Impact analysis on various sectors including technology, business, and society"
            ]
        },
        {
            "title": "Societal and Economic Transformation",
            "main_points": [
                "In-depth analysis of social implications and cultural changes",
                "Economic effects and business transformation across industries",
                "Workplace evolution and future of employment"
            ]
        },
        {
            "title": "Challenges, Solutions, and Ethical Considerations",
            "main_points": [
                "Major technical and implementation challenges",
                "Proposed solutions and recommendations",
                "Ethical implications and governance frameworks"
            ]
        }
    ]

    # 生成每个主要章节的内容
    for section in sections:
        print(f"\nGenerating section: {section['title']}")
        section_content = generate_comprehensive_section(main_topic, section)
        full_text += f"\n\n# {section['title']}\n\n{section_content}"

        # 增加一个章节总结和过渡
        summary_prompt = f"""Given this section about {section['title']}, provide a brief summary (about 500 words) that:
        1. Highlights key points
        2. Creates a smooth transition to the next section
        3. Reinforces main arguments
        Keep the summary focused and substantive."""

        summary_results, times = text_generation(summary_prompt + "\n\nContent: " + section_content, times)
        if summary_results:
            full_text += f"\n\n## Section Summary\n\n{summary_results.content}"

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