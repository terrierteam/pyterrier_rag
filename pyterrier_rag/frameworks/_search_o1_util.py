from outlines import prompt
from typing import List
import re

### PATTERNS ###

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

### PROMPT START ###


@prompt
def text_format(text: str, title: str = None) -> str:
    """
    {% if title != None %}
    Title: {{title}}
    {% endif %}
    Text: {{text}}
    """


def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


@prompt
def qwq_prompt(question: str):
    """Please answer the following question. You should provide your final answer in the format \\boxed{YOUR_ANSWER}.

    Question:\n{{ question }}\n\n
    """


@prompt
def generic_prompt(question: str):
    """Please answer the following question. You should think step by step to solve it.

    Provide your final answer in the format \\boxed{YOUR_ANSWER}.

    Question:\n{{ question }}\n\n
    """


@prompt
def get_webpage_to_reasonchain_instruction(prev_reasoning: str, search_query: str, document: str):
    """**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**
{{ prev_reasoning }}

- **Current Search Query:**
{{ search_query }}

- **Searched Web Pages:**
{{ document }}

Now you should analyze each web page and find helpful information based on the current search query "{{ search_query }}" and previous reasoning steps.
"""

### PROMPT END ###

### ANSWER EXTRACTION ###


def make_extract_answer(mode='gen'):
    def codegen_extract_answer(output):
        # Extract the code between ```python and ```
        extracted_text = ''
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
        return extracted_text

    def infogen_extract_answer(output):
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        extracted_text = ''
        pattern_info = "\n**Final Information**"
        pattern_step = "\n**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
        return extracted_text

    def gen_extract_answer(output):
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
        return extracted_text
    if mode == 'codegen':
        return codegen_extract_answer
    elif mode == 'infogen':
        return infogen_extract_answer
    else:
        return gen_extract_answer

### CONTEXT EDITING ###


def replace_recent_steps(origin_str, replace_str):

    """
    Replaces specific steps in the original reasoning steps with new steps.
    If a replacement step contains "DELETE THIS STEP", that step is removed.

    Parameters:
    - origin_str (str): The original reasoning steps.
    - replace_str (str): The steps to replace or delete.

    Returns:
    - str: The updated reasoning steps after applying replacements.
    """

    def parse_steps(text):
        """
        Parses the reasoning steps from a given text.

        Parameters:
        - text (str): The text containing reasoning steps.

        Returns:
        - dict: A dictionary mapping step numbers to their content.
        """
        step_pattern = re.compile(r"Step\s+(\d+):\s*")
        steps = {}
        current_step_num = None
        current_content = []

        for line in text.splitlines():
            step_match = step_pattern.match(line)
            if step_match:
                # If there's an ongoing step, save its content
                if current_step_num is not None:
                    steps[current_step_num] = "\n".join(current_content).strip()
                current_step_num = int(step_match.group(1))
                content = line[step_match.end():].strip()
                current_content = [content] if content else []
            else:
                if current_step_num is not None:
                    current_content.append(line)
            
        # Save the last step if any
        if current_step_num is not None:
            steps[current_step_num] = "\n".join(current_content).strip()
            
        return steps

    # Parse the original and replacement steps
    origin_steps = parse_steps(origin_str)
    replace_steps = parse_steps(replace_str)

    # Apply replacements
    for step_num, content in replace_steps.items():
        if "DELETE THIS STEP" in content:
            # Remove the step if it exists
            if step_num in origin_steps:
                del origin_steps[step_num]
        else:
            # Replace or add the step
            origin_steps[step_num] = content

    # Sort the steps by step number
    sorted_steps = sorted(origin_steps.items())

    # Reconstruct the reasoning steps as a single string
    new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

    return new_reasoning_steps

### GENERAL UTILITY ###


def extract_search_query(text: str) -> str:
    pattern = re.escape(BEGIN_SEARCH_QUERY) + r'(.*?)' + re.escape(END_SEARCH_QUERY)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def format_retrieval_docs(retrieval_results: List[List[dict]]) -> List[List[str]]:
    def truncate_text(text, max_words=360):
        words = text.split()
        if len(words) <= max_words:
            return text
        else:
            return " ".join(words[:max_words])

    retrieved_docs = [] 
    for retrieval_result in retrieval_results:
        docs = [] 
        for item in retrieval_result:
            title = item["title"] if "title" in item else None
            text = item["text"] if "text" in item else " ".join(sent.strip() for sent in item["sentences"])
            text = truncate_text(text)
            if title:
                docs.append(f"Title: {title}\nText: {text}")
            else:
                docs.append(f"Text: {text}")
        retrieved_docs.append(docs)
    return retrieved_docs
