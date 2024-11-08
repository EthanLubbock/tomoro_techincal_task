import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)

class StopAfterAnswerLine(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.answer_token_id = self.tokenizer.convert_tokens_to_ids("Answer:")
        self.newline_token_id = self.tokenizer.convert_tokens_to_ids("\n")
        self.answer_started = False

    def __call__(self, input_ids, scores, **kwargs):
        # Check if "Answer:" token has been generated
        if self.answer_token_id in input_ids[0]:
            self.answer_started = True
        
        # Stop when "Answer:" has been generated and a newline is detected after it
        if self.answer_started and self.newline_token_id in input_ids[0]:
            return True
        return False
    
class QueryAnswerModel:

    def __init__(self):
        # Load the fine-tuned model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained("llama-finetuned-convfinqa")
        self.tokenizer = AutoTokenizer.from_pretrained("llama-finetuned-convfinqa")

        if "Answer:" not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(["Answer:"])
            self.tokenizer.add_tokens(["\n"])
            self.model.resize_token_embeddings(len(self.tokenizer))

    def query_model(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")

        # Initialize custom stopping criteria with the "Answer:" token
        stopping_criteria = StoppingCriteriaList([StopAfterAnswerLine(self.tokenizer)])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria 
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer = answer.split("Answer:")[-1].strip()

        return answer

    @staticmethod
    def format_prompt(query, context, history):
        prompt = f"Context: {context}\nHistory:\n"
        if len(history) != 0:
            for qa_pair in history:
                q, a = qa_pair
                prompt += f"Q: {q}\nA: {a}\n"

        prompt += f"\nQuestion to answer: {query}\nAnswer: "
        return prompt