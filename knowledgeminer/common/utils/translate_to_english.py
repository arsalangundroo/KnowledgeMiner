from knowledgeminer.common.blocks.llm import azure_openai_for_llama_index
from llama_index.core.prompts.base import PromptTemplate
from knowledgeminer.prompts.prompt_templates import TRANSLATION_PROMPT_TEMPLATE


class OpenAITranslator(object):
    def __init__(self):
        self._openai_client = azure_openai_for_llama_index.create_basic_azure_openai_client()
    def tranlsate_german_to_english_with_gpt(self,original_text)->str:
        if not original_text or len(original_text.strip())==0:
            print("original text is empty !")
            return ""

        prompt_template = PromptTemplate(TRANSLATION_PROMPT_TEMPLATE)
        prompt=prompt_template.format(original_language="German", target_language = "English", original_text=original_text)
        response = self._openai_client.complete(prompt=prompt)
        translated_text = response.text.split("Translated_Text:")[1].strip()
        assert len(translated_text)>0
        return translated_text





if __name__=="__main__":
    translator = OpenAITranslator()
    res = translator.tranlsate_german_to_english_with_gpt("Nutzen Sie dieses Suchfeld, um eine FAQ innerhalb des OpenDXM Docupedia space zu finden Hier werden Sie die meist gestellten Fragen (FAQs) und die entsprechenden Antworten finden:")
    print(res)

