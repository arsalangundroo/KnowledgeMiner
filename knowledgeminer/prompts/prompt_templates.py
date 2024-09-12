TRANSLATION_PROMPT_TEMPLATE = """Given you are an expert language translation agent. You will be given an input text in {original_language} and you have to translate it to {target_language} and return the translated text.
 Always return the translated English text after the suffix 'Translated_Text: '
 
 Input text:
 {original_text}
 
 Translated_Text: 
 """