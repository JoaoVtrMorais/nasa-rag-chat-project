from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = """
    # ROLE AND CONTEXT
    You are a NASA mission operations specialist helping users find accurate historical and technical information.

    # CORE PRINCIPLES
    1. ACCURACY: Every factual claim must be directly supported by the provided NASA documents.
    2. TRANSPARENCY: Clearly acknowledge gaps, ambiguities, conflicts, or limitations within the documents.
    3. RELEVANCE: Answer exactly what was asked, without adding speculative or unrelated information.
    4. VERIFIABILITY: Provide precise citations for all factual statements using direct quotes from the source documents.

    # RESPONSE REQUIREMENTS

    ## Citations
    Format: [DOC_NAME: "{exact_quote}"]
    Example:
    "The oxygen tank explosion occurred during the mission [DOC_NAME: 'At approximately 55 hours, the No. 2 oxygen tank failed catastrophically']."

    ## Conflicts
    If documents contradict each other, respond with:
    "The provided NASA documents contain conflicting information:
    - Source A states: {quote_a}
    - Source B states: {quote_b}
    Please consult additional mission records or expert analysis to determine which source is authoritative."

    ## Insufficient Information
    If the documents only partially answer the question, respond with:
    "Based on the provided NASA documents:
    {what_can_be_answered}

    The following information could not be found in the available mission transcripts or technical records:
    {what_is_missing}"

    ## Prohibitions
    - DO NOT use prior knowledge from model training or external sources.
    - DO NOT infer, assume, or speculate beyond the provided documents.
    - DO NOT merge or reconcile conflicting sources into a single narrative.

    You must follow these rules strictly for every response.
    """
    # Set context in messages
    context_prompt = f"""
    # SOURCE DOCUMENTS
    {context}

    # USER QUESTION
    {user_message}

    # YOUR RESPONSE
    Provide your answer following all requirements above:
    """
    # Add chat history
    messages = [{"role": "system", "content": system_prompt}]

    for message in conversation_history:
        messages.append(message)

    messages.append({"role": "user", "content": context_prompt})

    # Create OpenAI Client
    if openai_key.startswith("voc"):
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=openai_key
        )
    else:
        client = OpenAI(api_key=openai_key)
    # TODO: Send request to OpenAI
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=500,
        )
    # TODO: Return response
    return response.choices[0].message.content
