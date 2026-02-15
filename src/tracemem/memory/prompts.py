SEGMENT_PROMPT = """
TASK:Topic Segmentation Analyzer and Semantic Memory Extractor

Input Example:
<D1>Jordan: Sam, have you tried that new Korean BBQ place on Main Street? Image: A modern Korean BBQ spot with a vibrant glass facade and prominent signage</D1>
<D2>Sam: Not yet, but my coworker said the bulgogi is amazing. I might go this weekend.</D2>
<D3>Jordan: Great! Are you still taking those photography classes?</D3>
<D4>Sam: Yes, I finished the beginner course last month. Now I'm learning portrait lighting techniques.</D4>
<D5>Jordan: That sounds interesting. By the way, I need to return some books to the library tomorrow.</D5>
<D6>Sam: Oh, I should go too. Actually, I'm thinking of getting a new laptop next week.</D6>
...

Output Template:
<Dn>
<intent>Follow DIALOGUE INTENT RULES</intent>
<semantic>Follow SEMANTIC MEMORY EXTRACTION RULES.</semantic>
</Dn>

DIALOGUE INTENT RULES
Determine Dn's intent by analyzing both Dn-1,Dn-2 and Dn+1,Dn+2 together.
1. CHANGE_TOPIC: Use this when the speaker introduces ANY new information, such as:
   - A new subject or activity, pay attention to keywords in the sentence.
   - A transition to a new domain or a "by the way" moment.
   - Prevent the situation where 10 consecutive texts accumulate without a topic change.
2. DEVELOP_TOPIC: Use this ONLY when the speaker is:
   - Just responding to the previous sentence.
   - Providing more detail about the exact same thing just mentioned without adding anything else.

SEMANTIC MEMORY EXTRACTION RULES:
1. Analyze both Dn-1 and Dn-2 to extract factual information. If Dn responds to Dn-1 or Dn-2 and introduces a new topic/entity, the semantic memory must reflect both aspects.
2. Focus on **WHO** **does/feels** **WHAT**, **WHEN**, **WHERE**.
3. Be exhaustive and detail-oriented. Do omit any information, any emotional adjectives, relative information or contextual nuances. 
4. DON'T OMIT any details from the image, including names, colors, and any other description. **Use format [Image: ...]**

Output Example:
<D1>
<intent>DEVELOP_TOPIC</intent>
<semantic>John asks Sam whether Sam has tried new Korean BBQ place on Main Street. Image: A modern Korean BBQ spot with a vibrant glass facade and prominent signage.</semantic>
</D1>
<D2>
<intent>DEVELOP_TOPIC</intent>
<semantic>Sam hasn't tried the new Korean BBQ restaurant. Sam's coworker recommends the bulgogi at the Korean BBQ restaurant. Sam might go to the Korean BBQ restaurant this weekend.</semantic>
</D2>
<D3>
<intent>CHANGE_TOPIC</intent>
<semantic></semantic>
</D3>
<D4>
<intent>DEVELOP_TOPIC</intent>
<semantic>Sam completed a beginner photography course last month. Sam is learning portrait lighting techniques.</semantic>
</D4>
<D5>
<intent>DEVELOP_TOPIC</intent>
<semantic>Jordan has books to return to the library tomorrow.</semantic>
</D5>
<D6>
<intent>CHANGE_TOPIC</intent>
<semantic>Sam also needs to go to the library and is considering purchasing a new laptop next week.</semantic>
</D6>
...
"""


EPISODE_PROMPT="""
TASK: Topic Summarizer

Input Template:
Current Time: <exact time>
person a: <text of person a>
person b: <text of person b>
...

EXAMPLE DIALOGUE:
Current Time: At 8:00 an on 10 December, 2020
Alex: Hey Sam! Have you finished the project report?
Sam: Not yet, but I'll complete it by Friday. Want to see the draft?
Alex: Yes, please send it over. By the way, did you hear about the team meeting?
Sam: No, what meeting?
Alex: There's a meeting tomorrow at 3 PM about the new client.
Sam: Thanks for telling me. I'll definitely attend.

EXAMPLE SUMMARY:
Summary: As of 10 December, 2020, Alex initiates contact by greeting Sam and inquiring whether the project report has been completed. Sam responds that it is not yet finished, but commits to completing it by Friday, offering to share a draft with Alex. Alex accepts this offer and requests that Sam send it over. Alex subsequently asks if Sam is aware of an upcoming team meeting, to which Sam asks for clarification. Alex informs Sam that the meeting is scheduled for the following day at 3 PM and concerns a new client. Sam confirms his attendance.
KEY RULES:
1. Must incorporate all keywords and topics discussed. 
2. **DO NOT OMIT any information**.Be exhaustive and detail-oriented, include but not limited to names, activity, attributes (e.g., colors, locations), and other essential details.
3. Describe in detail who did what at what point in time (if time mentions), or who did what with whom and when (if mentions). Do not omit any modifiers, adjectives, or other descriptive details.
4. If someone shows images, includes all the information of the image. Use format [Who] shows [What content].
5. Include **global time**(eg. As of 10 December, 2020) and any **relative time information** (eg. yesterday, last year).

Output Template:
...
"""

PERSONA_MODEL_PROMPT="""
TASK: Persona Slice Extraction

INPUT:
Speaker: ...
Episode Memory: ...
Labeled Semantic Memories: ...

Personal Experience Analysis:
   Experience: A chronological log of CONCRETE biographical facts about the character's OWN life.
   KEY REMINDER:
   - **What to Filter:** 
      - Exclude the information that is not revevant to the speaker's own life (eg. acknowledgments without information related to the speaker, sorely social pleasantries and greetings). MUST return "N/A" if no personal experiences are found.
   - **What to Extract:**
      - Extract ONLY information that belong in a person's own life (eg. possessions, actions, emotions, events, activities, habits, plans, routines, facts about themselves). 
      - Even if a personal fact is mentioned during an conversation topic about another person's experience, the **factual part about the speaker themselves MUST be extracted; 
      - Be exhaustive and detail-oriented. DON'T OMIT any specific information (eg. names, activities, colors, times).
      - Combine the **episode memory** and **semantic memories**, check for the speaker's personal experience carefully and extract them all. 
      - Must include image/photo details as original memories (e.g., names, colors). **Use format: Image: ...**
      - You MUST include the **global timestamp** and **any relative time information**.
      - You MUST copy and paste ALL original descriptors.

OUTPUT FORMAT:
{
  Experience": "As of..., [detailed experience description]" or "N/A"
}
"""

THEME_PROMPT = """
Task: Themes summarization.

INPUT CONTENT:
...

Requirements:
1. Title: Give a representative title.
2. Including as much raw keywords as possible.
4. DO NOT INCLUDE any time information.
5. ONLY one theme title.

OUTPUT FORMAT:
{
"theme_title": "..."
}
"""


TOPIC_PROMPT = """
Task: Topics summarization.

INPUT CONTENT:
...

Requirements:
1. Title: Give topic a representative title.
3. Including as much raw keywords as possible.
4. DO NOT INCLUDE any time information.
5. ONLY one topic title.

OUTPUT FORMAT:
{
"topic_title": "..."
}
"""

THREAD_PROMPT = """
Task: Threads summarization.

INPUT EXPERIENCES:
...

Requirements:
1. Title: Give thread a representative title.
2. Summary: Summarize the topics. Include all raw keywords (eg. names, times, colors, event, verbs).
3. Including all information in a concise way. Including as much raw keywords as possible.
4. DO NOT INCLUDE any time information.
5. ONLY one topic title and one summary.

OUTPUT FORMAT:
{
"thread_title": "...",
"summary": "..."
}
"""

USER_PROMPT="""
Task: Based on the question, choose one card/ two cards you need to search.

INPUT:
Question: [Insert question here]
Users in the conversation: <user a>, <user b>

RULES:
1. **STRICTLY choose ONLY from the users provided in the "Users in the conversation" input.**
2. If the question is only about <user a>, choose only <user a>. If only about <user b>, choose only <user b>.
3. If the question mentions both users or asks about their interaction, choose both.
4. **NEVER include any user name not listed in "Users in the conversation".**
5. **NEVER invent or assume additional users.**
6. Output MUST be valid JSON with ONLY the choice field (Users in the conversation: <user a>, <user b>).
7. The choice list must contain **at least one user name** from the provided list.

Example when Users in the conversation: "Amy", "Mike":
- Question: "What did Amy do yesterday?" → {"choice":["Amy"]}
- Question: "What did Mike and Amy discuss?" → {"choice":["Amy", "Mike"]}
- Question: "How are they feeling?" → {"choice":["Amy", "Mike"]}


OUTPUT FORMAT:
{
"choice":[**list of users, seperated by comma**]
}
"""


SEARCH_PROMPT="""
Task: Search threads.

INPUT(may have one or two cards):
Question:..
User Card (one or two):
user name 1: ...
<card of user 1>
user name 2: ...
<card of user 2>
OR
user name 1: ...
<card of user 1>

APPROACH:
1. First, read the question carefully before checking the card(s).
2. Check thread summaries under topics to find relevant threads.
3. Think step by step, give your reasoning, show your progress.
4. **Check OUTPUT FORMAT carefully**, return the output based on it.
5. **DO NOT give duplicate thread IDs**. Each thread_id should be correct and unique.
6. MUST return top relevant *10* threads per user with correct *thread_id*. **The result list under a person name cannot be empty.**
7. **CRITICAL: If input has two user cards, you MUST return results for BOTH users in the 'results' array.** 

**CRITICAL FORMAT RULES:**
- **Output MUST be pure, valid JSON without any comments, explanations, or extra text**
- **Do NOT include `//` comments in the JSON**
- **Do NOT include markdown code blocks (no ```json or ```)**
- **Start directly with `{` and end with `}`**

OUTPUT FORMAT:
Please output in the following **JSON format with no additional text**:

Format 1: If you receive two user cards:
{
  "reason": "...",
  "results": [
    {
      "PersonName1": [
        {"thread_id": "..."},
        {"thread_id": "..."},
        ...
      ]
    },
    {
      "PersonName2": [
        {"thread_id": "..."},
        {"thread_id": "..."},
        ...
      ]
    }
  ]
}

Format 2: If you receive one user card:
{
  "reason": "...",
  "results": [
    {
      "PersonName1": [
        {"thread_id": "..."},
        {"thread_id": "..."},
        ...
      ]
    }
  ]
}
"""

ANSWER_PROMPT="""
Task: Answer the question.

INPUT:
Question: [Insert question here]
Contents: ...

INSTRACTIONS:
1. If there is a question about time references (like "last year", "two months ago", etc.),
    calculate the actual date based on the memory timestamp. For example, if a memory from
    4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
2. Always convert relative time references to specific dates, months, or years. For example,
    convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
    timestamp. Ignore the reference while answering the question.
3. If the information only supports a broader timeframe (e.g. "a week before 3, July, 2024"), provide only that. 
4. Ensure your final answer is specific and avoids vague time references.
5. If the question asks about a specific event or fact, look for direct evidence. 
6. **Some of the questions may need your commonsense**, think carefully and **try your best guess**.
7. Some questions may require multiple evidences, link these evidences and reason.

APPROACH (Think step by step):
1. First, read the question and carefully before check the contents.
2. Then carefully analyze all provided contents. Pay attention to the information related with the question.
3. Carefully examine both absolute timestamps and relative time expressions in the contents. If the answer requires calculation (e.g., converting relative time references), **show you reason process**.
4. Please **think step by step very carefully** before providing the final answer, **show you reason process**.
5. When formulating your response, aim to use the wording and terms present in the contents, especially if it requests a description or statement. 

OUTPUT (include your reason and answer):
"...."
"""

