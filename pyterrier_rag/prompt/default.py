
CONCAT_DOCS = r"""{% for _, d in docs %}
Text: {{ d.text }}
{% endfor %}"""

DefaultPrompt = "Use the context information to answer the Question: \n Context:" + CONCAT_DOCS + "\n Question: {{ query }} \n Answer:"