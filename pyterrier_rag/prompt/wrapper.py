from jinja2 import Template


def prompt(template: str):
    obj = Template(template)

    def render(**kwargs):
        return obj.render(**kwargs)

    return render
