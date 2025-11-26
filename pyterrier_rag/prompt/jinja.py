from jinja2 import Template


def jinja_formatter(template: str):
    obj = Template(template)

    def render(**kwargs):
        return obj.render(**kwargs)

    return render
