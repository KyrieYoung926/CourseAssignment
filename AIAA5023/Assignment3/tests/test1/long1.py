from zhipuai import ZhipuAI
import json
import os
from typing import Tuple, List, Dict
import time

class LongTextGenerator:
    def __init__(self, api_key: str, topic: str, student_number: str):
        self.client = ZhipuAI(api_key=api_key)
        self.topic = topic
        self.student_number = student_number
        self.times = 0  # Count of API calls
        self.sections = []  # Store all generated sections
        self.outline = []  # Store article outline
        
    def save_progress(self, content: str, filename: str = "progress.json") -> None:
        """Save generation progress in case of interruption"""
        data = {
            "topic": self.topic,
            "times": self.times,
            "sections": self.sections,
            "outline": self.outline,
            "last_content": content
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_progress(self, filename: str = "progress.json") -> bool:
        """Load previous generation progress"""
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.topic = data["topic"]
                self.times = data["times"]
                self.sections = data["sections"]
                self.outline = data["outline"]
            return True
        return False

    def generate_outline(self) -> List[str]:
        """Generate article outline"""
        prompt = f"""As a professional writer, please create a detailed outline for the topic "{self.topic}". Requirements:
1. Include at least 8 main sections
2. Each section should have 3-4 subsections
3. Ensure logical flow and clear hierarchy
4. Plan key points and transitions for each section
Return only the outline content, no additional explanation."""
        
        response, _ = self._call_api(prompt)
        self.outline = response.content.split("\n")
        return self.outline

    def generate_section(self, section_title: str, prev_content: str = "") -> str:
        """Generate content for a single section"""
        context = f"Previous content summary: {prev_content[:200]}..." if prev_content else "This is the beginning of the article"
        prompt = f"""Based on the topic "{self.topic}", generate detailed content for section "{section_title}". Requirements:
1. Content should be detailed, at least 1500 words
2. Ensure coherence with context: {context}
3. Use specific examples and data to support arguments
4. Ensure natural transitions between paragraphs
Return only the main content, no additional explanation."""
        
        response, _ = self._call_api(prompt)
        return response.content

    def polish_section(self, content: str) -> str:
        """Polish and improve section content"""
        prompt = f"""Please polish and refine the following content to ensure:
1. More elegant and fluid language
2. Clearer logic
3. Enhanced details and examples
4. Improved transition sentences
Content:
{content}"""
        
        response, _ = self._call_api(prompt)
        return response.content

    def _call_api(self, prompt: str, max_retries: int = 3) -> Tuple[object, int]:
        """Wrapper for API calls with retry mechanism"""
        for i in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[{"role": "user", "content": prompt}]
                )
                self.times += 1
                return response.choices[0].message, self.times
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                time.sleep(2)  # Wait 2 seconds before retry

    def generate_full_text(self) -> str:
        """Generate complete long-form text"""
        # 1. Check for previous progress
        if not self.load_progress():
            # 2. Generate outline
            self.generate_outline()
            self.save_progress("")
        
        # 3. Generate content for each section based on outline
        prev_content = ""
        for i, section in enumerate(self.outline):
            if i < len(self.sections):  # Skip already generated sections
                prev_content = self.sections[i]
                continue
                
            if not section.strip():  # Skip empty lines
                continue
                
            # Generate new content
            content = self.generate_section(section, prev_content)
            polished_content = self.polish_section(content)
            self.sections.append(polished_content)
            prev_content = polished_content
            
            # Save progress
            self.save_progress(polished_content)
            
            # Add sleep to avoid API limits
            time.sleep(1)
        
        # 4. Merge all content
        final_text = "\n\n".join(self.sections)
        
        # 5. Save final result
        self.save_final_text(final_text)
        
        return final_text

    def save_final_text(self, text: str) -> None:
        """Save final generated text"""
        filename = f"{self.student_number}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

    def get_statistics(self) -> Dict:
        """Get statistics about generated text"""
        final_text = "\n\n".join(self.sections)
        return {
            "total_words": len(final_text),
            "sections_count": len(self.sections),
            "api_calls": self.times
        }

# Usage example
if __name__ == "__main__":
    TOPIC = "A Futuristic Cafe"  # Choose one from the three given topics
    API_KEY = "6d5a58010e2bf9e564b2aaea8c233627.5IAcpKe37uRdXiBu"
    STUDENT_NUMBER = "21094327"
    
    generator = LongTextGenerator(API_KEY, TOPIC, STUDENT_NUMBER)
    final_text = generator.generate_full_text()
    stats = generator.get_statistics()
    
    print(f"Generation completed!")
    print(f"Total words: {stats['total_words']}")
    print(f"Number of sections: {stats['sections_count']}")
    print(f"Total API calls: {stats['api_calls']}")