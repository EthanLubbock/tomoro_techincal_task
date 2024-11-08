from datasets import DatasetInfo, GeneratorBasedBuilder, Split, SplitGenerator, Value, Features, Sequence

class ConvFinQA(GeneratorBasedBuilder):
    VERSION = "1.0.1"
    
    def _info(self):
        return DatasetInfo(
            description="""
                ConvFinQA is a conversational financial QA dataset from EMNLP 2022 paper:

                ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering
            """,
            features=Features({
                "context": Value("string"),
                "questions": Sequence(Value("string")),
                "answers": Sequence(Value("string"))
            }),
            supervised_keys=None,
            homepage="https://arxiv.org/abs/2210.03849",
            citation="""@article{chen2022convfinqa,
                title={ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering},
                author={Chen, Zhiyu and Li, Shiyang and Smiley, Charese and Ma, Zhiqiang and Shah, Sameena and Wang, William Yang},
                journal={Proceedings of EMNLP 2022},
                year={2022}
            }"""
        )

    def _split_generators(self, dl_manager):
        # Paths to data files
        train_path = "data/train.json"
        validation_path = "data/validation.json"

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"filepath": train_path}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"filepath": validation_path})
        ]

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""
        import json
        
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for idx, entry in enumerate(data):
                yield idx, {
                    "context": entry["context"],
                    "questions": entry["questions"],
                    "answers": entry["answers"]
                }