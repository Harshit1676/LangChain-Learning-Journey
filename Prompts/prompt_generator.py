from langchain_core.prompts import PromptTemplate
from langchain_core.load import dumps
from pathlib import Path

template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
    template="""
    Please explain the research paper titled {paper_input}
    in a {style_input} style.
    The explanation should be {length_input} in length.
    """
)

Path("template.json").write_text(dumps(template), encoding="utf-8")
